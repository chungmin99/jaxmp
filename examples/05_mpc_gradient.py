"""05_mpc.py
Run sampling- or gradient-based MPC in collision aware environments.
"""

from typing import Optional
from pathlib import Path
import time
import jax

from loguru import logger
import tyro

import jax.numpy as jnp
import jaxlie
import numpy as onp

import viser
import viser.extras

from jaxmp import JaxKinTree, RobotFactors
from jaxmp.coll import Plane, RobotColl, CollGeom, link_to_spheres, Capsule
from jaxmp.extras import load_urdf, solve_mpc


def main(
    robot_description: str = "panda",
    robot_urdf_path: Optional[Path] = None,
    n_steps: int = 5,
    dt: float = 0.1,
    use_world_collision: bool = True,
    use_self_collision: bool = False,
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
    # sphere_obs = Sphere.from_center_and_radius(jnp.zeros(3), jnp.array([0.05]))
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

    JointVar = RobotFactors.get_var_class(kin)

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
        joint_traj, cost = solve_mpc(
            kin,
            robot_coll,
            [] if not use_world_collision else [ground_obs, curr_sphere_obs],
            target_poses,
            target_joint_indices,
            joints,
            JointVar,
            n_steps=n_steps,
            dt=dt,
            prev_sols=joint_traj,
            use_self_collision=use_self_collision,
            use_world_collision=use_world_collision,
        )
        jax.block_until_ready((joint_traj, cost))
        timing_handle.value = (time.time() - start) * 1000

        cost_handle.value = cost.item()
        joints = joint_traj[0]

        # Update timing info.
        if not has_jitted:
            logger.info("JIT compile + runing took {} ms.", timing_handle.value)
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
