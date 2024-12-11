# Write MPC, using `jaxmp`.


from typing import Optional, Literal
from pathlib import Path
import time
import jax
import jaxls

from jaxmp.coll._collide_types import make_frame
from loguru import logger
import tyro

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp

import viser
import viser.extras

from jaxmp import JaxKinTree, RobotFactors
from jaxmp.coll import Plane, RobotColl, Sphere, CollGeom, link_to_spheres, Capsule
from jaxmp.extras._urdf_loader import load_urdf

def ik_cost(
    vals, kin, joint_var, pose_var, weights, target_joint_idx
):
    joint_cfg = vals[joint_var]
    target_pose = vals[pose_var]
    Ts_joint_world = kin.forward_kinematics(joint_cfg)
    residual = (
        (
            jaxlie.SE3(Ts_joint_world[target_joint_idx])
        ).inverse()
        @ (target_pose)
    ).log()
    weights = jnp.broadcast_to(weights, residual.shape)
    assert residual.shape == weights.shape
    return (residual * weights).flatten()


@jdc.jit
def solve_mpc(
    kin: JaxKinTree,
    coll: RobotColl,
    world_coll: CollGeom,
    goal_pose: jaxlie.SE3,
    goal_joint_idx: jax.Array,
    n_steps: int = 5,
    dt: float = 0.1,
    start_joints: Optional[jnp.ndarray] = None,
    pos_weight: float = 5.0,
    rot_weight: float = 2.0,
):
    """
    Formulate MPC problem as a series of x and u optimization variables.
    - x: joint angles
    - u: joint velocities

    Goal is to:
    - Minimize distance to goal pose, at each timestep, and
    - Ensure that all constraints are satisfied:
        - Joint limits
        - Velocity limits
        - Collision avoidance
    """

    if start_joints is None:
        start_joints = (kin.limits_upper + kin.limits_lower) / 2
    assert start_joints.shape == (kin.num_actuated_joints,)

    JointVar = RobotFactors.get_var_class(kin, start_joints)

    factors = []
    start_pose = jaxlie.SE3(kin.forward_kinematics(start_joints)[..., goal_joint_idx, :])
    for i in range(n_steps):
        factors.append(
            RobotFactors.ik_cost_factor(
                JointVar,
                i,
                kin,
                goal_pose,
                goal_joint_idx,
                jnp.array([pos_weight] * 3 + [rot_weight] * 3),
            )
            # jaxls.Factor(
            #     ik_cost,
            #     (
            #         kin,
            #         JointVar(i),
            #         jaxls.SE3Var(i),
            #         jnp.array([pos_weight] * 3 + [rot_weight] * 3),
            #         goal_joint_idx,
            #     )
            # )
        )
        # if i == 0:
        #     factors.append(
        #         jaxls.Factor(
        #             lambda vals, var: (vals[var].inverse() @ start_pose).log().flatten()
        #             * jnp.array([pos_weight] * 3 + [rot_weight] * 3),
        #             (jaxls.SE3Var(i),),
        #         )
        #     )
        # else:
        #     factors.append(
        #         jaxls.Factor(
        #             lambda vals, var: (vals[var].inverse() @ goal_pose).log().flatten()
        #             * jnp.array([pos_weight] * 3 + [rot_weight] * 3),
        #             (jaxls.SE3Var(i),),
        #         )
        #     )
        factors.append(
            RobotFactors.limit_cost_factor(
                JointVar, i, kin, jnp.array([100.0] * kin.num_actuated_joints)
            )
        )
        factors.append(
            RobotFactors.limit_vel_cost_factor(
                JointVar,
                i,
                kin,
                dt,
                jnp.array([10.0] * kin.num_actuated_joints),
                prev_var_idx=(None if i == 0 else (i - 1)),
                prev_cfg=(start_joints if i == 0 else None),
            )
        )
        # Convert these to sweeps.
        # factors.extend(
        #     RobotFactors.self_coll_factors(
        #         JointVar,
        #         i,
        #         kin,
        #         coll,
        #         0.05,
        #         5.0,
        #     )
        # )
        factors.extend(
            RobotFactors.manipulability_cost_factors(
            JointVar,
            i,
            kin,
            goal_joint_idx,
            0.01,
            )
        )
        if i > 0:
            # factors.append(
            #     RobotFactors.smoothness_cost_factor(
            #         JointVar, i, i - 1, jnp.array([0.1] * kin.num_actuated_joints)
            #     )
            # )
            factors.extend(
                RobotFactors.world_coll_factors(
                    JointVar,
                    i,
                    kin,
                    coll,
                    world_coll,
                    0.05,
                    20.0,
                    prev_var_idx=(i - 1),
                )
            )

    joint_vars = [JointVar(i) for i in range(n_steps)]

    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars,
        use_onp=False,
    )
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(joint_vars),
        # initial_vals=jaxls.VarValues.make(
        #     [jv.with_value(start_joints) for jv in joint_vars[:n_steps]] + 
        #     [jv.with_value(start_pose) for jv in joint_vars[n_steps:]]
        # ),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        verbose=False,
    )

    joints = jnp.stack([solution[jv] for jv in joint_vars[:n_steps]])
    return joints


def main(
    robot_description: str = "panda",
    robot_urdf_path: Optional[Path] = None,
):
    urdf = load_urdf(robot_description, robot_urdf_path)
    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2
    coll = RobotColl.from_urdf(urdf, create_coll_bodies=link_to_spheres)

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    urdf_vis.update_cfg(onp.array(rest_pose))
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    add_joint_button = server.gui.add_button("Add joint!")
    target_name_handles: list[viser.GuiDropdownHandle] = []
    target_tf_handles: list[viser.TransformControlsHandle] = []
    target_frame_handles: list[viser.BatchedAxesHandle] = []
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    def add_joint():
        idx = len(target_name_handles)
        target_name_handle = server.gui.add_dropdown(
            f"target joint {idx}",
            list(urdf.joint_names),
            initial_value=urdf.joint_names[0],
        )
        target_tf_handle = server.scene.add_transform_controls(
            f"target_transform_{idx}", scale=0.2
        )
        target_frame_handle = server.scene.add_batched_axes(
            f"target_{idx}",
            axes_length=0.5 * 0.2,
            axes_radius=0.05 * 0.2,
            batched_positions=onp.broadcast_to(onp.array([0.0, 0.0, 0.0]), (10, 3)),
            batched_wxyzs=onp.broadcast_to(onp.array([1.0, 0.0, 0.0, 0.0]), (10, 4)),
        )

        target_name_handles.append(target_name_handle)
        target_tf_handles.append(target_tf_handle)
        target_frame_handles.append(target_frame_handle)

    add_joint_button.on_click(lambda _: add_joint())
    add_joint()

    # sphere_obs = Sphere.from_center_and_radius(jnp.zeros(3), jnp.array([0.05]))
    sphere_obs = Capsule.from_radius_and_height(
        radius=jnp.array([0.05]), 
        height=jnp.array([2.0]), 
        transform=jaxlie.SE3.from_translation(jnp.zeros(3))
    )
    sphere_obs_handle = server.scene.add_transform_controls(
        "sphere_obs", scale=0.2, position=(0.2, 0.0, 0.2)
    )
    server.scene.add_mesh_trimesh("sphere_obs/mesh", sphere_obs.to_trimesh())

    joints = rest_pose
    has_jitted = False
    while True:
        if len(target_name_handles) == 0:
            time.sleep(0.1)
            continue

        goal_joint_idx = jnp.array(
            [
                kin.joint_names.index(target_name_handle.value)
                for target_name_handle in target_name_handles
            ]
        )
        goal_pose_list = [
            jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))
            for target_tf_handle in target_tf_handles
        ]

        goal_pose = jaxlie.SE3(jnp.stack([pose.wxyz_xyz for pose in goal_pose_list]))

        world_coll = sphere_obs.transform(
            jaxlie.SE3(
                jnp.array([*sphere_obs_handle.wxyz, *sphere_obs_handle.position])
            )
        )

        start = time.time()
        joint_traj = solve_mpc(kin, coll, world_coll, goal_pose, goal_joint_idx, start_joints=joints)
        jax.block_until_ready(joint_traj)
        timing_handle.value = (time.time() - start) * 1000
        if not has_jitted:
            logger.info("JIT compile + running took {} ms.", timing_handle.value)
            has_jitted = True

        joints = joint_traj[1] + 0.2 * (joint_traj[1] - joints)
        joint_traj = joint_traj.at[0].set(joints)
        urdf_vis.update_cfg(onp.array(joints))

        for target_frame_handle, target_joint_idx in zip(
            target_frame_handles, goal_joint_idx
        ):
            T_target_world = jaxlie.SE3(kin.forward_kinematics(joint_traj)[..., target_joint_idx, :])
            target_frame_handle.positions_batched = onp.array(T_target_world.translation())
            target_frame_handle.wxyzs_batched = onp.array(T_target_world.rotation().wxyz)


if __name__ == "__main__":
    tyro.cli(main)
