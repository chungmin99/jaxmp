"""01_kinematics.py
Tests robot forward + inverse kinematics using JaxMP.
"""

from typing import Literal, Optional
from pathlib import Path
import time
from loguru import logger
import tyro
import viser
import viser.extras

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp

from jaxmp import JaxKinTree, RobotFactors
from jaxmp.extras.urdf_loader import load_urdf
from jaxmp.extras.solve_ik import solve_ik

try:
    import sksparse
except ImportError:
    logger.info("sksparse not found. Some solvers may not work.")
    sksparse = None


def main(
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    device: Literal["cpu", "gpu"] = "cpu",
    robot_description: Optional[str] = "yumi",
    robot_urdf_path: Optional[Path] = None,
):
    """
    Test robot inverse kinematics using JaxMP.
    Args:
        pos_weight: Weight for position error in IK.
        rot_weight: Weight for rotation error in IK.
        rest_weight: Weight for rest pose in IK.
        limit_weight: Weight for joint limits in IK.
        device: Device to use.
        robot_description: Name of the robot description to load.
        robot_urdf_path: Path to the robot URDF file.
    """
    # Set device.
    jax.config.update("jax_platform_name", device)

    # Load robot description.
    urdf = load_urdf(robot_description, robot_urdf_path)

    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Add base-frame freezing logic.
    with server.gui.add_folder("Base frame"):
        freeze_base_x = server.gui.add_checkbox("Freeze x", initial_value=True)
        freeze_base_y = server.gui.add_checkbox("Freeze y", initial_value=True)
        freeze_base_z = server.gui.add_checkbox("Freeze z", initial_value=True)
        freeze_base_rx = server.gui.add_checkbox("Freeze rx", initial_value=True)
        freeze_base_ry = server.gui.add_checkbox("Freeze ry", initial_value=True)
        freeze_base_rz = server.gui.add_checkbox("Freeze rz", initial_value=True)

    def get_freeze_base_xyz_xyz() -> jnp.ndarray:
        return jnp.array(
            [
                freeze_base_x.value,
                freeze_base_y.value,
                freeze_base_z.value,
                freeze_base_rx.value,
                freeze_base_ry.value,
                freeze_base_rz.value,
            ]
        ).astype(jnp.float32)

    # Add base-frame freezing logic.
    with server.gui.add_folder("Target frame"):
        freeze_target_x = server.gui.add_checkbox("Freeze x", initial_value=True)
        freeze_target_y = server.gui.add_checkbox("Freeze y", initial_value=True)
        freeze_target_z = server.gui.add_checkbox("Freeze z", initial_value=True)
        freeze_target_rx = server.gui.add_checkbox("Freeze rx", initial_value=True)
        freeze_target_ry = server.gui.add_checkbox("Freeze ry", initial_value=True)
        freeze_target_rz = server.gui.add_checkbox("Freeze rz", initial_value=True)

    def get_freeze_target_xyz_xyz() -> jnp.ndarray:
        return jnp.array(
            [
                freeze_target_x.value,
                freeze_target_y.value,
                freeze_target_z.value,
                freeze_target_rx.value,
                freeze_target_ry.value,
                freeze_target_rz.value,
            ]
        ).astype(jnp.float32)

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    tf_size_handle = server.gui.add_slider(
        "Gizmo size", min=0.01, max=0.4, step=0.01, initial_value=0.2
    )

    if sksparse is None:
        solver_types = ("conjugate_gradient", "dense_cholesky")
    else:
        solver_types = ("conjugate_gradient", "dense_cholesky", "cholmod")
    solver_type_handle = server.gui.add_dropdown(
        "Solver type",
        solver_types,
        initial_value="conjugate_gradient",
    )

    smooth_handle = server.gui.add_checkbox("Smooth", initial_value=False)

    with server.gui.add_folder("Manipulability"):
        manipulabiltiy_weight_handler = server.gui.add_slider(
            "weight", 0.0, 0.01, 0.001, 0.00
        )
        manipulability_cost_handler = server.gui.add_number(
            "Yoshikawa index", 0.001, disabled=True
        )

    set_frames_to_current_pose = server.gui.add_button("Set frames to current pose")
    add_joint_button = server.gui.add_button("Add joint!")

    target_name_handles: list[viser.GuiDropdownHandle] = []
    target_tf_handles: list[viser.TransformControlsHandle] = []
    target_frame_handles: list[viser.FrameHandle] = []

    # Put robot to rest pose :-)
    base_pose = jaxlie.SE3.identity()
    joints = rest_pose

    urdf_base_frame.position = onp.array(base_pose.translation())
    urdf_base_frame.wxyz = onp.array(base_pose.rotation().wxyz)
    urdf_vis.update_cfg(onp.array(joints))

    # Add joints.
    def add_joint():
        idx = len(target_name_handles)
        target_name_handle = server.gui.add_dropdown(
            f"target joint {idx}",
            list(urdf.joint_names),
            initial_value=urdf.joint_names[0],
        )
        target_tf_handle = server.scene.add_transform_controls(
            f"target_transform_{idx}", scale=tf_size_handle.value
        )
        target_frame_handle = server.scene.add_frame(
            f"target_{idx}",
            axes_length=0.5 * tf_size_handle.value,
            axes_radius=0.05 * tf_size_handle.value,
            origin_radius=0.1 * tf_size_handle.value,
        )
        target_name_handles.append(target_name_handle)
        target_tf_handles.append(target_tf_handle)
        target_frame_handles.append(target_frame_handle)

    add_joint_button.on_click(lambda _: add_joint())
    add_joint()

    # Let the user change the size of the transformcontrol gizmo.
    @tf_size_handle.on_update
    def _(_):
        for target_tf_handle in target_tf_handles:
            target_tf_handle.scale = tf_size_handle.value
        for target_frame_handle in target_frame_handles:
            target_frame_handle.axes_length = 0.5 * tf_size_handle.value
            target_frame_handle.axes_radius = 0.05 * tf_size_handle.value
            target_frame_handle.origin_radius = 0.1 * tf_size_handle.value

    # Set target frames to where it is on the currently displayed robot.
    # We need to put them in world frame (since our goal is to match joint-to-world).
    @set_frames_to_current_pose.on_click
    def _(_):
        nonlocal joints
        base_pose = jnp.array(
            urdf_base_frame.wxyz.tolist() + urdf_base_frame.position.tolist()
        )

        for target_frame_handle, target_name_handle, target_tf_handle in zip(
            target_frame_handles, target_name_handles, target_tf_handles
        ):
            target_joint_idx = kin.joint_names.index(target_name_handle.value)
            T_target_world = jaxlie.SE3(base_pose) @ jaxlie.SE3(
                kin.forward_kinematics(joints)[target_joint_idx]
            )

            target_frame_handle.position = onp.array(T_target_world.translation())
            target_frame_handle.wxyz = onp.array(T_target_world.rotation().wxyz)
            target_tf_handle.position = onp.array(T_target_world.translation())
            target_tf_handle.wxyz = onp.array(T_target_world.rotation().wxyz)

    has_jitted = False
    while True:
        # Don't do anything if there are no target joints...
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
        manipulability_weight = manipulabiltiy_weight_handler.value

        if smooth_handle.value:
            initial_pose = joints
            joint_vel_weight = limit_weight
        else:
            initial_pose = rest_pose
            joint_vel_weight = 0.0

        # Solve!
        start_time = time.time()
        base_pose, joints = solve_ik(
            kin,
            target_poses,
            target_joint_indices,
            initial_pose,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            rest_weight=rest_weight,
            limit_weight=limit_weight,
            manipulability_weight=manipulability_weight,
            include_manipulability=(manipulability_weight > 0),
            joint_vel_weight=joint_vel_weight,
            solver_type=solver_type_handle.value,
            freeze_base_xyz_xyz=get_freeze_target_xyz_xyz(),
            freeze_target_xyz_xyz=get_freeze_base_xyz_xyz(),
        )

        # Ensure all computations are complete before measuring time
        jax.block_until_ready((base_pose, joints))
        timing_handle.value = (time.time() - start_time) * 1000
        if not has_jitted:
            logger.info("JIT compile + runing took {} ms.", timing_handle.value)
            has_jitted = True

        # Update visualizations.
        urdf_base_frame.position = onp.array(base_pose.translation())
        urdf_base_frame.wxyz = onp.array(base_pose.rotation().wxyz)
        urdf_vis.update_cfg(onp.array(joints))
        for target_frame_handle, target_joint_idx in zip(
            target_frame_handles, target_joint_indices
        ):
            T_target_world = base_pose @ jaxlie.SE3(
                kin.forward_kinematics(joints)[target_joint_idx]
            )
            target_frame_handle.position = onp.array(T_target_world.translation())
            target_frame_handle.wxyz = onp.array(T_target_world.rotation().wxyz)

        # Update manipulability cost.
        manip_cost = 0
        for target_joint_idx in target_joint_indices:
            manip_cost += RobotFactors.manip_yoshikawa(kin, joints, target_joint_idx)
        manip_cost /= len(target_joint_indices)
        manipulability_cost_handler.value = onp.array(manip_cost).item()


if __name__ == "__main__":
    tyro.cli(main)
