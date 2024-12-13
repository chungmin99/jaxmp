"""06_mpc_sampling.py
Run sampling-based MPC using MPPI in collision aware environments.
"""

from typing import Optional
from pathlib import Path
import time
import jax
from loguru import logger
import tyro
import jaxlie
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as onp
import viser
import viser.extras
from jaxmp import JaxKinTree, RobotFactors
from jaxmp.coll import Plane, RobotColl, CollGeom, link_to_spheres, Capsule
from jaxmp.extras import load_urdf


@jdc.jit
def mppi(
    kin: JaxKinTree,
    robot_coll: RobotColl,
    world_coll_list: list[CollGeom],
    target_pose: jaxlie.SE3,
    target_joint_indices: jax.Array,
    initial_joints: jnp.ndarray,
    n_steps: jdc.Static[int],
    dt: float,
    rest_pose: jnp.ndarray,
    n_samples: int = 10000,
    lambda_: float = 0.1,
    noise_sigma: float = 0.005,
    gamma: float = 0.9,
    *,
    pos_weight: float = 5.0,
    rot_weight: float = 2.0,
    limit_weight: float = 100.0,
    joint_vel_weight: float = 10.0,
    joint_smoothness_weight: float = 0.1,
    pose_smoothness_weight: float = 1.0,
    use_self_collision: bool = False,
    self_collision_weight: float = 20.0,
    use_world_collision: bool = False,
    world_collision_weight: float = 20.0,
) -> jnp.ndarray:
    """
    Perform MPPI to find the optimal joint trajectory.
    """
    def cost_function(traj):
        # Control actions -> joint trajectory.
        joint_cfg = jnp.cumsum(traj, axis=0) + initial_joints
        
        # Define cost, discount.
        cost = jnp.zeros(joint_cfg.shape[0])
        discount = gamma ** jnp.arange(n_steps)

        # Joint limit cost.
        residual_upper = jnp.maximum(0.0, joint_cfg - kin.limits_upper) * limit_weight
        residual_lower = jnp.maximum(0.0, kin.limits_lower - joint_cfg) * limit_weight
        residual = (residual_upper + residual_lower).sum(axis=-1)
        cost += residual

        # Joint velocity limit cost.
        cost = cost.at[1:].add(
            jnp.maximum(
                0.0, jnp.abs(jnp.diff(joint_cfg, axis=0)) - kin.joint_vel_limit * dt
            ).sum(axis=1)
            * joint_vel_weight
        )

        # EE pose cost.
        Ts_joint_world = kin.forward_kinematics(joint_cfg)
        residual = (
            (jaxlie.SE3(Ts_joint_world[..., target_joint_indices, :])).inverse()
            @ target_pose
        ).log() * jnp.array([pos_weight] * 3 + [rot_weight] * 3)
        cost += jnp.abs(residual).sum(axis=(-1, -2))

        # Joint smoothness cost.
        cost += 

        # Manipulability cost
        # manipulability = jax.vmap(RobotFactors.manip_yoshikawa, in_axes=(None, 0, None))(kin, joint_cfg, target_joint_indices).sum(axis=-1)
        # cost += jnp.where(manipulability < 0.05, 1.0 - manipulability, 0.0) * 0.1

        # # Slight bias towards zero config
        # cost += jnp.linalg.norm(joint_cfg - rest_pose, axis=-1) * 0.01

        cost = cost * discount
        assert cost.shape == (joint_cfg.shape[0],)
        return cost

    def sample_trajectories(mean_trajectory, covariance, key):
        noise = jax.random.multivariate_normal(
            key, mean=jnp.zeros(kin.num_actuated_joints), cov=covariance, shape=(n_samples, n_steps)
        )
        return mean_trajectory[None, :, :] + noise

    key = jax.random.PRNGKey(0)

    mean_trajectory = jnp.broadcast_to(jnp.zeros_like(initial_joints), (n_steps, kin.num_actuated_joints))
    covariance = jnp.eye(kin.num_actuated_joints) * 0.005**2

    key, subkey = jax.random.split(key)
    sampled_trajectories = sample_trajectories(mean_trajectory, covariance, subkey)
    costs = jax.vmap(cost_function)(sampled_trajectories)
    weights = jnp.exp(-costs / lambda_)
    weights /= jnp.sum(weights, axis=0)
    alpha = 1.0  # Temperature parameter
    mean_trajectory = (1 - alpha) * mean_trajectory + alpha * jnp.sum(weights[..., None] * sampled_trajectories, axis=0)
    covariance = (1 - alpha) * covariance + alpha * jax.vmap(lambda x: jnp.cov(x, rowvar=False), in_axes=1)(sampled_trajectories * weights[..., None] - mean_trajectory)

    return jnp.cumsum(mean_trajectory, axis=0) + initial_joints


def main(
    robot_description: str = "panda",
    robot_urdf_path: Optional[Path] = None,
    n_steps: int = 20,
    dt: float = 0.1,
    use_world_collision: bool = True,
    use_self_collision: bool = False,
):
    urdf = load_urdf(robot_description, robot_urdf_path)
    robot_coll = RobotColl.from_urdf(urdf, create_coll_bodies=link_to_spheres)
    kin = JaxKinTree.from_urdf(urdf, unroll_fk=True)
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
    ground_obs_handle = server.scene.add_mesh_trimesh(
        "ground_plane", ground_obs.to_trimesh()
    )
    server.scene.add_grid(
        "ground", width=3, height=3, cell_size=0.1, position=(0.0, 0.0, 0.001)
    )

    # Also add a movable sphere as an obstacle (world collision).
    sphere_obs = Capsule.from_radius_and_height(
        radius=jnp.array([0.05]),
        height=jnp.array([2.0]),
        transform=jaxlie.SE3.from_translation(jnp.zeros(3)),
    )
    sphere_obs_handle = server.scene.add_transform_controls(
        "sphere_obs", scale=0.2, position=(0.2, 0.0, 0.2)
    )
    server.scene.add_mesh_trimesh("sphere_obs/mesh", sphere_obs.to_trimesh())
    if not use_world_collision:
        sphere_obs_handle.visible = False
        ground_obs_handle.visible = False

    # Add GUI elements, to let user interact with the robot joints.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    cost_handle = server.gui.add_number("cost", 0.01, disabled=True)
    add_joint_button = server.gui.add_button("Add joint!")
    target_name_handles: list[viser.GuiDropdownHandle] = []
    target_tf_handles: list[viser.TransformControlsHandle] = []
    target_frame_handles: list[viser.BatchedAxesHandle] = []

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
        target_frame_handle = server.scene.add_batched_axes(
            f"target_{idx}",
            axes_length=0.05,
            axes_radius=0.005,
            batched_positions=onp.broadcast_to(
                onp.array([0.0, 0.0, 0.0]), (n_steps, 3)
            ),
            batched_wxyzs=onp.broadcast_to(
                onp.array([1.0, 0.0, 0.0, 0.0]), (n_steps, 4)
            ),
        )
        target_name_handles.append(target_name_handle)
        target_tf_handles.append(target_tf_handle)
        target_frame_handles.append(target_frame_handle)

    add_joint_button.on_click(lambda _: add_joint())
    add_joint()

    joints = rest_pose
    joint_traj = jnp.broadcast_to(rest_pose, (n_steps, kin.num_actuated_joints))



    has_jitted = False
    while True:
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
        joint_traj = mppi(
            kin,
            robot_coll,
            [] if not use_world_collision else [ground_obs, curr_sphere_obs],
            target_poses,
            target_joint_indices,
            joints,
            n_steps=n_steps,
            dt=dt,
            use_self_collision=use_self_collision,
            use_world_collision=use_world_collision,
            rest_pose=rest_pose,
        )
        jax.block_until_ready(joint_traj)
        timing_handle.value = (time.time() - start) * 1000

        if jnp.isnan(joint_traj).any():
            continue

        cost_handle.value = 0.0  # MPPI does not return cost directly
        joints = joint_traj[0]

        # Update timing info.
        if not has_jitted:
            logger.info("JIT compile + running took {} ms.", timing_handle.value)
            has_jitted = True

        urdf_vis.update_cfg(onp.array(joints))

        for target_frame_handle, target_joint_idx in zip(
            target_frame_handles, target_joint_indices
        ):
            T_target_world = jaxlie.SE3(
                kin.forward_kinematics(joint_traj)[..., target_joint_idx, :]
            )
            target_frame_handle.positions_batched = onp.array(
                T_target_world.translation()
            )
            target_frame_handle.wxyzs_batched = onp.array(
                T_target_world.rotation().wxyz
            )


if __name__ == "__main__":
    tyro.cli(main)
