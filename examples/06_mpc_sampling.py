"""06_mpc_sampling.py
Run sampling-based MPC using MPPI in collision aware environments.

Poorly tuned, but still works / shows the idea.
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
    lambda_: float = 0.2,
    noise_sigma: float = 0.02,
    gamma: float = 0.9,
    *,
    pos_weight: float = 5.0,
    rot_weight: float = 2.0,
    limit_weight: float = 100.0,
    joint_vel_weight: float = 10.0,
    use_world_collision: jdc.Static[bool] = False,
    world_collision_weight: float = 20.0,
    n_iterations: jdc.Static[int] = 5,
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

        # Manipulability cost
        manipulability = jax.vmap(
            RobotFactors.manip_yoshikawa, in_axes=(None, 0, None)
        )(kin, joint_cfg, target_joint_indices).sum(axis=-1)
        cost += jnp.where(manipulability < 0.05, 1.0 - manipulability, 0.0) * 0.1

        # # Slight bias towards zero config
        cost += jnp.linalg.norm(joint_cfg - rest_pose, axis=-1) * 0.01

        # World collision cost.
        # if dist < 0, then we are in collision.
        if use_world_collision:
            for world_coll in world_coll_list:
                world_coll_dist = robot_coll.world_coll_dist(kin, joint_cfg, world_coll)
                cost += jnp.clip(-world_coll_dist, min=0.0) * world_collision_weight

        cost = cost * discount
        assert cost.shape == (joint_cfg.shape[0],)
        return cost

    def mppi_iteration(mean_trajectory, covariance, key):
        # Shape assertions for inputs
        assert mean_trajectory.shape == (
            n_steps,
            kin.num_actuated_joints,
        ), f"mean_trajectory shape: {mean_trajectory.shape}, expected: ({n_steps}, {kin.num_actuated_joints})"
        assert covariance.shape == (
            kin.num_actuated_joints,
            kin.num_actuated_joints,
        ), f"covariance shape: {covariance.shape}, expected: ({kin.num_actuated_joints}, {kin.num_actuated_joints})"

        # Sample trajectories: (n_samples, n_steps, n_joints)
        noise = jax.random.multivariate_normal(
            key,
            mean=jnp.zeros(kin.num_actuated_joints),
            cov=covariance,
            shape=(n_samples, n_steps),
        )
        assert noise.shape == (
            n_samples,
            n_steps,
            kin.num_actuated_joints,
        ), f"noise shape: {noise.shape}"

        sampled_trajectories = (
            mean_trajectory[None, :, :] + noise
        )  # (n_samples, n_steps, n_joints)
        assert sampled_trajectories.shape == (
            n_samples,
            n_steps,
            kin.num_actuated_joints,
        ), f"sampled_trajectories shape: {sampled_trajectories.shape}"

        # Evaluate costs per timestep: (n_samples, n_steps)
        costs = jax.vmap(cost_function)(sampled_trajectories)  # (n_samples, n_steps)
        assert costs.shape == (
            n_samples,
            n_steps,
        ), f"costs shape: {costs.shape}, expected: ({n_samples}, {n_steps})"

        # Calculate weights per timestep
        weights = jnp.exp(-costs / lambda_)  # (n_samples, n_steps)
        weights = weights / (
            jnp.sum(weights, axis=0)[None, :] + 1e-8
        )  # Normalize per timestep
        assert weights.shape == (n_samples, n_steps), f"weights shape: {weights.shape}"

        # Update mean trajectory per timestep
        weights_expanded = weights[:, :, None]  # (n_samples, n_steps, 1)
        new_mean = jnp.sum(
            weights_expanded * sampled_trajectories, axis=0
        )  # (n_steps, n_joints)
        assert new_mean.shape == (
            n_steps,
            kin.num_actuated_joints,
        ), f"new_mean shape: {new_mean.shape}"

        # Update covariance per timestep
        centered = (
            sampled_trajectories - new_mean[None, :, :]
        )  # (n_samples, n_steps, n_joints)
        new_covariance = jnp.zeros_like(covariance)

        def compute_weighted_covariance(t):
            weighted_centered = (
                weights[:, t, None] * centered[:, t, :]
            )  # (n_samples, n_joints)
            return jnp.einsum("ni,nj->ij", weighted_centered, centered[:, t, :])

        new_covariance += jax.vmap(compute_weighted_covariance)(
            jnp.arange(n_steps)
        ).sum(axis=0)
        new_covariance = (
            new_covariance / n_steps + jnp.eye(kin.num_actuated_joints) * 1e-4
        )

        return new_mean, new_covariance

    # Initialize
    key = jax.random.PRNGKey(0)
    mean_trajectory = jnp.zeros((n_steps, kin.num_actuated_joints))
    covariance = jnp.eye(kin.num_actuated_joints) * noise_sigma**2

    # Run multiple iterations
    def scan_fn(carry, _):
        mean_trajectory, covariance, key = carry
        key, subkey = jax.random.split(key)
        mean_trajectory, covariance = mppi_iteration(
            mean_trajectory, covariance, subkey
        )
        return (mean_trajectory, covariance, key), None

    (mean_trajectory, covariance, _), _ = jax.lax.scan(
        scan_fn, (mean_trajectory, covariance, key), None, length=n_iterations
    )

    return jnp.cumsum(mean_trajectory, axis=0) + initial_joints


def main(
    robot_description: str = "panda",
    robot_urdf_path: Optional[Path] = None,
    n_steps: int = 20,
    dt: float = 0.1,
    use_world_collision: bool = True,
):
    urdf = load_urdf(robot_description, robot_urdf_path)
    robot_coll = RobotColl.from_urdf(urdf, create_coll_bodies=link_to_spheres)
    kin = JaxKinTree.from_urdf(urdf, unroll_fk=True)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2
    assert isinstance(robot_coll.coll, CollGeom)

    server = viser.ViserServer(port=8081)

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

    mppi_params = {
        "lambda": server.gui.add_slider(
            "Lambda", min=0.01, max=1.0, initial_value=0.2, step=0.01
        ),
        "noise_sigma": server.gui.add_slider(
            "Noise Sigma", min=0.001, max=0.1, initial_value=0.02, step=0.001
        ),
        "gamma": server.gui.add_slider(
            "Gamma (discount)", min=0.1, max=1.0, initial_value=0.9, step=0.01
        ),
        "n_iterations": server.gui.add_slider(
            "Number of iterations", min=1, max=10, initial_value=1, step=1
        ),
    }

    # Add GUI elements for weights
    weight_params = {
        "pos_weight": server.gui.add_slider(
            "Position Weight", min=0.1, max=20.0, initial_value=5.0, step=0.1
        ),
        "rot_weight": server.gui.add_slider(
            "Rotation Weight", min=0.1, max=20.0, initial_value=2.0, step=0.1
        ),
        "world_collision_weight": server.gui.add_slider(
            "Collision Weight", min=0.1, max=50.0, initial_value=20.0, step=0.1
        ),
        "limit_weight": server.gui.add_slider(
            "Joint Limit Weight", min=1.0, max=200.0, initial_value=100.0, step=1.0
        ),
        "joint_vel_weight": server.gui.add_slider(
            "Joint Velocity Weight", min=0.1, max=50.0, initial_value=10.0, step=0.1
        ),
    }

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
            use_world_collision=use_world_collision,
            rest_pose=rest_pose,
            lambda_=mppi_params["lambda"].value,
            noise_sigma=mppi_params["noise_sigma"].value,
            gamma=mppi_params["gamma"].value,
            pos_weight=weight_params["pos_weight"].value,
            rot_weight=weight_params["rot_weight"].value,
            world_collision_weight=weight_params["world_collision_weight"].value,
            limit_weight=weight_params["limit_weight"].value,
            joint_vel_weight=weight_params["joint_vel_weight"].value,
            n_iterations=mppi_params["n_iterations"].value,
        )
        jax.block_until_ready(joint_traj)
        timing_handle.value = (time.time() - start) * 1000

        if jnp.isnan(joint_traj).any():
            continue

        cost_handle.value = 0.0  # MPPI does not return cost directly
        joints = joint_traj[1]

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
