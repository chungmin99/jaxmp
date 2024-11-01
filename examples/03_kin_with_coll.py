"""03_kin_with_coll.py
Similar to 01_kinematics.py, but with collision avoidance as a cost.
"""

from typing import Optional, Literal
from pathlib import Path
import time
import jax
import jaxls
from jaxmp.coll import collide

from loguru import logger
import tyro

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp

import viser
import viser.extras

from jaxmp import JaxKinTree, RobotFactors
from jaxmp.coll import Plane, RobotColl, Sphere, CollGeom
from jaxmp.extras.urdf_loader import load_urdf


def main(
    robot_description: str = "yumi",
    robot_urdf_path: Optional[Path] = None,
    device: Literal["cpu", "gpu"] = "cpu",
):
    # Set device.
    jax.config.update("jax_platform_name", device)

    urdf = load_urdf(robot_description, robot_urdf_path)
    robot_coll = RobotColl.from_urdf(urdf)
    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2
    assert isinstance(robot_coll.coll, CollGeom)

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    urdf_vis.update_cfg(onp.array(rest_pose))
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Create ground plane as an obstacle (world collision)!
    ground_obs = Plane.from_point_and_normal(
        jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    )
    server.scene.add_mesh_trimesh("ground_plane", ground_obs.to_trimesh())
    server.scene.add_grid(
        "ground", width=3, height=3, cell_size=0.1, position=(0.0, 0.0, 0.001)
    )

    # Also add a movable sphere as an obstacle (world collision).
    sphere_obs = Sphere.from_center_and_radius(jnp.zeros(3), jnp.array([0.05]))
    sphere_obs_handle = server.scene.add_transform_controls(
        "sphere_obs", scale=0.2, position=(0.2, 0.0, 0.2)
    )
    server.scene.add_mesh_trimesh("sphere_obs/mesh", sphere_obs.to_trimesh())

    # Visualize collision distances.
    self_coll_value = server.gui.add_number(
        "max. coll (self)", 0.0, step=0.01, disabled=True
    )
    world_coll_value = server.gui.add_number(
        "max. coll (world)", 0.0, step=0.01, disabled=True
    )
    visualize_coll = server.gui.add_checkbox("Visualize spheres", False)

    # Add GUI elements, to let user interact with the robot joints.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    add_joint_button = server.gui.add_button("Add joint!")
    target_name_handles: list[viser.GuiDropdownHandle] = []
    target_tf_handles: list[viser.TransformControlsHandle] = []
    target_frame_handles: list[viser.FrameHandle] = []

    def add_joint():
        # Show target joint name.
        idx = len(target_name_handles)
        target_name_handle = server.gui.add_dropdown(
            f"target joint {idx}",
            list(urdf.joint_names),
            initial_value=urdf.joint_names[0],
        )
        target_tf_handle = server.scene.add_transform_controls(
            f"target_transform_{idx}", scale=0.2
        )
        target_frame_handle = server.scene.add_frame(f"target_{idx}", axes_length=0.1)

        target_name_handles.append(target_name_handle)
        target_tf_handles.append(target_tf_handle)
        target_frame_handles.append(target_frame_handle)

    add_joint_button.on_click(lambda _: add_joint())
    add_joint()

    joints = rest_pose

    # Create factor graph.
    collbody_handle = None

    has_jitted = False
    while True:
        if visualize_coll.value:
            assert isinstance(robot_coll.coll, CollGeom)
            collbody_handle = server.scene.add_mesh_trimesh(
                "coll",
                robot_coll.coll.transform(
                    jaxlie.SE3(
                        kin.forward_kinematics(joints)[
                            ..., robot_coll.link_joint_idx, :
                        ]
                    )
                ).to_trimesh(),
            )
        elif collbody_handle is not None:
            collbody_handle.remove()

        if len(target_name_handles) == 0:
            time.sleep(0.1)
            continue

        target_joint_indices = jnp.array(
            [
                kin.joint_names.index(target_name_handle.value)
                for target_name_handle in target_name_handles
            ]
        )
        target_pose_list = [
            jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))
            for target_tf_handle in target_tf_handles
        ]

        target_poses = jaxlie.SE3(
            jnp.stack([pose.wxyz_xyz for pose in target_pose_list])
        )

        curr_sphere_obs = sphere_obs.transform(
            jaxlie.SE3(
                jnp.array([*sphere_obs_handle.wxyz, *sphere_obs_handle.position])
            )
        )

        start = time.time()
        joints = solve_ik_with_coll(
            kin,
            target_joint_indices,
            target_poses,
            robot_coll,
            [ground_obs, curr_sphere_obs],
            joints,
        )
        jax.block_until_ready(joints)
        # Update timing info.
        timing_handle.value = (time.time() - start) * 1000
        if not has_jitted:
            logger.info("JIT compile + runing took {} ms.", timing_handle.value)
            has_jitted = True

        urdf_vis.update_cfg(onp.array(joints))

        for target_frame_handle, target_joint_idx in zip(
            target_frame_handles, target_joint_indices
        ):
            T_target_world = kin.forward_kinematics(joints)[target_joint_idx]
            target_frame_handle.position = onp.array(T_target_world)[4:]
            target_frame_handle.wxyz = onp.array(T_target_world)[:4]

        urdf_vis.update_cfg(onp.array(joints))

        coll = robot_coll.at_joints(kin, joints)
        assert isinstance(coll, CollGeom)
        dist = collide(coll, coll.reshape(-1, 1)).dist
        coll_list = jnp.array(robot_coll.self_coll_list)
        dist = dist[coll_list[:, 0], coll_list[:, 1]]
        self_coll_value.value = dist.min().item()

        world_coll_value.value = min(
            collide(coll, ground_obs).dist.min().item(),
            collide(coll, curr_sphere_obs).dist.min().item(),
        )


@jdc.jit
def solve_ik_with_coll(
    kin: JaxKinTree,
    target_joint_indices: jnp.ndarray,
    target_poses: jaxlie.SE3,
    robot_coll: RobotColl,
    world_coll: list[CollGeom],
    initial_pose: jnp.ndarray,
    *,
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    self_coll_weight: float = 5.0,
    world_coll_weight: float = 10.0,
) -> jnp.ndarray:
    # Create factor graph.
    factors: list[jaxls.Factor] = []

    JointVar = RobotFactors.get_var_class(kin, initial_pose)
    joint_var_idx = 0

    ik_weight = jnp.array([pos_weight] * 3 + [rot_weight] * 3)
    factors.extend(
        [
            RobotFactors.ik_cost_factor(
                JointVar,
                joint_var_idx,
                kin,
                target_poses,
                target_joint_indices,
                ik_weight,
            ),
            RobotFactors.rest_cost_factor(
                JointVar,
                joint_var_idx,
                jnp.array([rest_weight] * kin.num_actuated_joints),
            ),
            RobotFactors.limit_vel_cost_factor(
                JointVar,
                joint_var_idx,
                kin,
                0.01,
                jnp.array([limit_weight] * kin.num_actuated_joints),
                initial_pose,
            ),
            RobotFactors.limit_cost_factor(
                JointVar,
                joint_var_idx,
                kin,
                jnp.array([limit_weight] * kin.num_actuated_joints),
            ),
        ]
    )

    # Add collision factors.
    self_coll_factors = RobotFactors.self_coll_factors(
        JointVar, joint_var_idx, kin, robot_coll, 0.05, self_coll_weight
    )
    world_coll_factors = [
        RobotFactors.get_world_coll_factors(
            JointVar, joint_var_idx, kin, robot_coll, coll, 0.1, world_coll_weight
        )
        for coll in world_coll
    ]

    factors.extend(self_coll_factors)
    for world_coll_factor in world_coll_factors:
        factors.extend(world_coll_factor)

    # Solve IK.
    joint_vars = [JointVar(joint_var_idx)]
    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars,
        use_onp=False,
    )
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(joint_vars),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5,
            parameter_tolerance=1e-5,
            max_iterations=50,
        ),
        verbose=False,
    )

    # Update visualization.
    joints = solution[JointVar(joint_var_idx)]
    return joints


if __name__ == "__main__":
    tyro.cli(main)
