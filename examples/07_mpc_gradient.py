"""01_mpc_gradient.py
Migrated from jaxmp example: Run gradient-based MPC in collision aware environments.
"""

from typing import Optional, Sequence
from pathlib import Path
import time
import jax

from loguru import logger
import tyro

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
from jaxtyping import Array, Float, Int

import viser
import viser.extras

import pyroki as pk
from pyroki.coll import RobotCollision, Capsule, HalfSpace, CollGeom


@jdc.jit
def run_mpc_step(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    n_steps: jdc.Static[int],
    w_pos: jdc.Static[float],
    w_rot: jdc.Static[float],
    w_limit: jdc.Static[float],
    w_limit_vel: jdc.Static[float],
    w_world_coll: jdc.Static[float],
    w_smoothness: jdc.Static[float],
    collision_margin: jdc.Static[float],
    initial_joints: Float[Array, " actuated_count"],
    prev_sols: Optional[Float[Array, " n_steps actuated_count"]],
    target_poses: jaxlie.SE3,
    target_link_indices: Int[Array, " num_targets"],
    current_obstacles: Sequence[CollGeom],
    dt: jdc.Static[float],
) -> tuple[Float[Array, " n_steps actuated_count"], Float[Array, ""]]:
    """Performs one step of MPC: setup, solve, and return results. JIT-compiled."""

    assert target_poses.translation().shape[0] == len(target_link_indices)

    num_targets = target_poses.translation().shape[0]

    def batched_rplus(
        pose: jaxlie.SE3,
        delta: jax.Array,
    ) -> jaxlie.SE3:
        return jax.vmap(jaxlie.manifold.rplus)(pose, delta.reshape(num_targets, -1))

    # Custom SE3 variable to batch across multiple joint targets.
    # This is not to be confused with SE3Vars with ids, which we use here for timesteps.
    class BatchedSE3Var(  # pylint: disable=missing-class-docstring
        pk.optim.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.identity((num_targets,)),
        retract_fn=batched_rplus,
        tangent_dim=jaxlie.SE3.tangent_dim * num_targets,
    ): ...

    class MatchJointToPoseCost(pk.CostFactor[pk.optim.Var[jnp.ndarray], BatchedSE3Var]):
        def cost_fn(self, vals, joint_var, pose_var):
            joint_cfg = vals[joint_var]
            target_pose = vals[pose_var]
            Ts_joint_world = robot.forward_kinematics(joint_cfg)
            residual = (
                (jaxlie.SE3(Ts_joint_world[..., target_link_indices, :])).inverse()
                @ (target_pose)
            ).log()
            return residual

    class SE3SmoothnessCost(pk.CostFactor[BatchedSE3Var, BatchedSE3Var]):
        def cost_fn(self, vals, pose_var, pose_var_prev):
            return (vals[pose_var].inverse() @ vals[pose_var_prev]).log().flatten()

    class SE3PoseMatchCost(pk.CostFactor[BatchedSE3Var]):
        def cost_fn(self, vals, pose_var):
            return (vals[pose_var].inverse() @ target_poses).log().flatten()

    factors: list[pk.CostFactor] = [
        MatchJointToPoseCost(
            cost_inputs=(
                robot.JointVar(jnp.arange(0, n_steps)),
                BatchedSE3Var(jnp.arange(0, n_steps)),
            ),
            weights=100.0,
        ),
        SE3SmoothnessCost(
            cost_inputs=(
                BatchedSE3Var(jnp.arange(0, n_steps)),
                BatchedSE3Var(jnp.arange(1, n_steps + 1)),
            ),
            weights=1.0,
        ),
        SE3PoseMatchCost(
            cost_inputs=(BatchedSE3Var(n_steps - 1),),
            weights=jnp.array([w_pos] * 3 + [w_rot] * 3),
        ),
    ]

    factors.extend([
        pk.SmoothnessCost(
            cost_inputs=(
                robot.JointVar(jnp.arange(0, n_steps - 1)),
                robot.JointVar(jnp.arange(1, n_steps)),
            ),
            weights=w_smoothness,
        ),
        pk.LimitVelCost(
            cost_inputs=(
                robot.JointVar(jnp.arange(0, n_steps - 1)),
                robot.JointVar(jnp.arange(1, n_steps)),
            ),
            weights=w_limit_vel,
            robot=robot,
            dt=dt,
        ),
        pk.LimitCost(
            cost_inputs=(robot.JointVar(jnp.arange(0, n_steps)),),
            weights=w_limit,
            robot=robot,
        ),
        pk.ManipulabilityCost(
            cost_inputs=(robot.JointVar(jnp.arange(0, n_steps)),),
            weights=0.001,
            robot=robot,
            target_link_indices=target_link_indices,
        ),
    ])

    factors.extend(
        [
            pk.WorldCollisionCost(
                cost_inputs=(robot.JointVar(jnp.arange(0, n_steps)),),
                weights=w_world_coll,
                robot=robot,
                robot_coll=robot_coll,
                world_geom=obs,
                margin=collision_margin,
            )
            for obs in current_obstacles
        ]
    )

    traj_var = robot.JointVar(jnp.arange(0, n_steps))
    pose_var = BatchedSE3Var(jnp.arange(0, n_steps))

    if prev_sols is not None:
        init_traj_vals = prev_sols
    else:
        init_traj_vals = jnp.repeat(initial_joints, n_steps, axis=0)

    init_pose_vals = jaxlie.SE3(robot.forward_kinematics(init_traj_vals)[..., target_link_indices, :])

    init_var = traj_var.with_value(init_traj_vals)
    init_pose_var = pose_var.with_value(init_pose_vals)

    solution, cost_vector = pk.solve(
        [traj_var, pose_var],
        factors,
        init_vars=[init_var, init_pose_var],
    )
    cost = jnp.sum(cost_vector**2)
    joint_traj = solution[traj_var]

    return joint_traj, cost


def main(
    robot_description: str = "panda",
    robot_urdf_path: Optional[Path] = None,
    n_steps: int = 5,
    dt: float = 0.02,
    w_pos: float = 5.0,
    w_rot: float = 2.0,
    w_limit: float = 100.0,
    w_limit_vel: float = 10.0,
    w_self_coll: float = 20.0,
    w_world_coll: float = 10.0,
    w_smoothness: float = 10.0,
    collision_margin: float = 0.01,
):
    """Main execution function."""

    urdf, robot = pk.load_robot(
        robot_urdf_path=robot_urdf_path, robot_description=robot_description
    )
    robot_coll = RobotCollision.from_urdf(urdf)
    rest_pose = (robot.joint.upper_limits_act + robot.joint.lower_limits_act) / 2

    server = viser.ViserServer()
    viser_robot = pk.viewer.BatchedURDF(server, urdf=urdf, root_node_name="/robot")
    viser_robot.update_cfg(rest_pose)

    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Use HalfSpace for ground plane
    ground_obs = HalfSpace.from_point_and_normal(
        jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    )
    server.scene.add_grid(
        "ground_grid", width=3, height=3, cell_size=0.1, position=(0.0, 0.0, 0.001)
    )

    capsule_obs_initial = Capsule.from_center_radius_height(
        center=jnp.array([0.0, 0.0, 0.0]),
        orientation_mat=jnp.eye(3),
        radius=jnp.array(0.05),
        height=jnp.array(1.0),
    )
    capsule_tf_handle = server.scene.add_transform_controls(
        "/capsule_obs", scale=0.2, position=(0.2, 0.0, 0.2)
    )
    server.scene.add_mesh_trimesh("/capsule_obs/mesh", capsule_obs_initial.to_trimesh())

    current_capsule_geom = capsule_obs_initial

    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    cost_handle = server.gui.add_number("Cost", 0.01, disabled=True)
    add_target_button = server.gui.add_button("Add Target Link")
    target_name_handles: list[viser.GuiDropdownHandle] = []
    target_tf_handles: list[viser.TransformControlsHandle] = []
    target_frame_handles: list[viser.BatchedAxesHandle] = []

    MAX_TARGETS = 1

    def add_target_link():
        if len(target_name_handles) >= MAX_TARGETS:
            logger.warning(f"Max targets ({MAX_TARGETS}) reached.")
            return

        idx = len(target_name_handles)
        target_name_handle = server.gui.add_dropdown(
            f"target link {idx}",
            list(robot.link.names),
            initial_value=robot.link.names[-1],
        )
        target_tf_handle = server.scene.add_transform_controls(
            f"target_transform_{idx}", scale=0.2, position=(0.5, 0.0, 0.5)
        )
        target_frame_handle: viser.BatchedAxesHandle = server.scene.add_batched_axes(
            f"target_{idx}",
            axes_length=0.05,
            axes_radius=0.005,
            batched_positions=onp.zeros((n_steps, 3)),
            batched_wxyzs=onp.array([[1.0, 0.0, 0.0, 0.0]] * n_steps),
        )
        target_name_handles.append(target_name_handle)
        target_tf_handles.append(target_tf_handle)
        target_frame_handles.append(target_frame_handle)

    add_target_button.on_click(lambda _: add_target_link())
    add_target_link()

    joints = rest_pose
    joint_traj = jnp.broadcast_to(rest_pose, (n_steps, robot.joint.actuated_count))

    has_jitted_solve = False
    while True:
        if len(target_name_handles) == 0:
            logger.warning("Waiting for at least one target link...")
            time.sleep(0.5)
            continue

        # Prepare dynamic inputs with fixed shapes for JIT
        link_name_to_idx = {name: i for i, name in enumerate(robot.link.names)}
        active_target_indices = [
            link_name_to_idx[target_name_handle.value]
            for target_name_handle in target_name_handles
        ]
        active_target_poses = [
            jaxlie.SE3.from_rotation_and_translation(
                rotation=jaxlie.SO3(jnp.asarray(target_tf_handle.wxyz)),
                translation=jnp.asarray(target_tf_handle.position),
            )
            for target_tf_handle in target_tf_handles
        ]

        num_active_targets = len(active_target_indices)
        padded_target_indices = jnp.array(
            active_target_indices + [-1] * (MAX_TARGETS - num_active_targets)
        )
        stacked_poses_wxyz_xyz = jnp.stack(
            [p.wxyz_xyz for p in active_target_poses]
            + [jaxlie.SE3.identity().wxyz_xyz] * (MAX_TARGETS - num_active_targets)
        )
        padded_target_poses = jaxlie.SE3(stacked_poses_wxyz_xyz)

        current_target_indices = padded_target_indices[:num_active_targets]
        current_target_poses = jaxlie.SE3(
            padded_target_poses.wxyz_xyz[:num_active_targets]
        )
        if num_active_targets == 0:
            current_target_indices = jnp.array([-1])
            current_target_poses = jaxlie.SE3.identity()

        # Prepare obstacles list
        current_obstacles_list: list[CollGeom] = [ground_obs]
        T_world_capsule = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(jnp.asarray(capsule_tf_handle.wxyz)),
            translation=jnp.asarray(capsule_tf_handle.position),
        )
        current_obstacles_list.append(current_capsule_geom.transform(T_world_capsule))

        start_time = time.time()

        # Call the combined JIT-compiled step function
        joint_traj, cost = run_mpc_step(
            robot=robot,
            robot_coll=robot_coll,
            n_steps=n_steps,
            w_pos=w_pos,
            w_rot=w_rot,
            w_limit=w_limit,
            w_limit_vel=w_limit_vel,
            w_world_coll=w_world_coll,
            w_smoothness=w_smoothness,
            collision_margin=collision_margin,
            initial_joints=joints,
            prev_sols=joint_traj,
            target_poses=current_target_poses,
            target_link_indices=current_target_indices,
            current_obstacles=current_obstacles_list,
            dt=dt,
        )
        jax.block_until_ready(joint_traj)

        timing_handle.value = (time.time() - start_time) * 1000
        cost_handle.value = float(cost)

        joints = joint_traj[0]

        if not has_jitted_solve:
            logger.info("First solve step (incl. JIT) took {} ms.", timing_handle.value)
            has_jitted_solve = True

        viser_robot.update_cfg(joints)

        link_poses_traj = robot.forward_kinematics(joint_traj)
        for i, target_frame_handle in enumerate(target_frame_handles):
            if i < num_active_targets:
                target_link_idx = active_target_indices[i]
                T_target_world_traj = jaxlie.SE3(link_poses_traj[:, target_link_idx, :])
                target_frame_handle.visible = True
                target_frame_handle.batched_positions = onp.array(
                    T_target_world_traj.translation()
                )
                target_frame_handle.batched_wxyzs = onp.array(
                    T_target_world_traj.rotation().wxyz
                )
            else:
                target_frame_handle.visible = False


if __name__ == "__main__":
    tyro.cli(main)
