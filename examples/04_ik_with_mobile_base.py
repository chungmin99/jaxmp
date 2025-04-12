"""04_ik_with_mobile_base.py
Inverse Kinematics with an Optimizable Mobile Base using PyRoKi.
"""

import time
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import tyro
import viser
from loguru import logger

import pyroki as pk
from pyroki.viewer import BatchedURDF


@jdc.jit
def solve_ik(
    robot: pk.Robot,
    target_pose: jaxlie.SE3,
    target_link_indices: jnp.ndarray,
    base_constraint_flags: jnp.ndarray,
    init_joints: Optional[jnp.ndarray],
    init_base_pose: Optional[jaxlie.SE3],
    *,
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    base_constraint_cost_weight: float = 1.0,
    max_iterations: int = 1,
) -> Tuple[jaxlie.SE3, jax.Array]:
    """Solves the mobile base IK problem using penalty-based constraints.

    Args:
        robot: The Pyroki Robot model.
        target_pose: Desired SE(3) pose for target link(s) in world frame.
        target_link_indices: Indices of the links to target.
        base_constraint_flags: Flags (shape 6) penalizing base pose deviation.
        init_joints: Initial guess for joint configuration.
        init_base_pose: Initial guess for base pose.
        pos_weight: Weight for position error.
        rot_weight: Weight for rotation error.
        rest_weight: Weight for rest pose deviation.
        limit_weight: Weight for joint limits proximity.
        base_constraint_cost_weight: Overall weight for the base constraint factor.
        max_iterations: Max solver iterations.

    Returns:
        Tuple of (optimized base pose, optimized joint configuration).
    """

    def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
        delta = delta * (1 - base_constraint_flags)
        return pk.optim.SE3Var.retract_fn(transform, delta)

    class ConstrainedSE3Var(
        pk.optim.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.identity(),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=retract_fn,
    ): ...

    base_pose_var = ConstrainedSE3Var(0)
    joint_var = robot.JointVar(0)
    vars = [joint_var, base_pose_var]

    # Determine the rest pose: use init_joints if provided, else default
    if init_joints is not None:
        rest_pose_for_cost = init_joints
    else:
        rest_pose_for_cost = joint_var.default_factory()

    factors = [
        pk.PoseCostWithBase(
            (
                joint_var,
                base_pose_var,  # Pass the variable instance
                target_pose,
            ),
            robot=robot,
            target_link_indices=target_link_indices,
            weights=jnp.array([pos_weight] * 3 + [rot_weight] * 3),
        ),
        pk.LimitCost(
            (joint_var,),
            robot=robot,
            weights=jnp.array([limit_weight] * robot.joint.count),
        ),
        pk.RestCostWithBase(
            (
                joint_var,
                base_pose_var,
            ),
            rest_pose=rest_pose_for_cost,  # Pass the determined rest pose
            weights=jnp.array(
                [rest_weight] * robot.joint.actuated_count
                + [base_constraint_cost_weight] * 6
            ),
        ),
    ]

    init_vars = []
    if init_joints is not None:
        init_vars.append(joint_var.with_value(init_joints))
    else:
        init_vars.append(joint_var)

    if init_base_pose is not None:
        init_vars.append(base_pose_var.with_value(init_base_pose))
    else:
        init_vars.append(base_pose_var)

    sol, _ = pk.solve(vars, factors, init_vars=init_vars, max_iterations=max_iterations)
    return sol[base_pose_var], sol[joint_var]


def setup_robot(
    robot_description: Optional[str], robot_urdf_path: Optional[Path]
) -> Tuple[Any, pk.Robot]:
    """Loads the robot model."""
    logger.info("Loading robot model...")
    urdf, robot = pk.load_robot(
        robot_urdf_path=robot_urdf_path, robot_description=robot_description
    )
    logger.info(
        "Loaded robot '{}' with {} joints ({} actuated) and {} links.".format(
            robot_description or robot_urdf_path,
            robot.joint.count,
            robot.joint.actuated_count,
            robot.link.count,
        )
    )
    return urdf, robot


def setup_visualization_and_gui(
    server: viser.ViserServer,
    urdf: Any,
    robot: pk.Robot,
) -> Tuple[BatchedURDF, Dict[str, Any]]:
    """Initializes the Viser visualizer and GUI elements."""
    server.scene.configure_default_lights()

    gui_handles = {}
    gui_handles["base_frame"] = server.scene.add_frame(
        "/base", show_axes=True, axes_length=0.2
    )
    urdf_vis = BatchedURDF(server, urdf, root_node_name="/base")
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    gui_handles["timing"] = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    gui_handles["smooth"] = server.gui.add_checkbox("DiffIK", initial_value=False)

    with server.gui.add_folder("Base Constraints"):
        gui_handles["base_constraint_x"] = server.gui.add_checkbox("Fix X", True)
        gui_handles["base_constraint_y"] = server.gui.add_checkbox("Fix Y", True)
        gui_handles["base_constraint_z"] = server.gui.add_checkbox("Fix Z", True)
        gui_handles["base_constraint_rx"] = server.gui.add_checkbox("Fix RX", True)
        gui_handles["base_constraint_ry"] = server.gui.add_checkbox("Fix RY", True)
        gui_handles["base_constraint_rz"] = server.gui.add_checkbox("Fix RZ", True)

    with server.gui.add_folder("Cost weights"):
        gui_handles["pos_weight"] = server.gui.add_slider(
            "Position", 0.0, 50.0, 0.1, 5.0
        )
        gui_handles["rot_weight"] = server.gui.add_slider(
            "Rotation", 0.0, 10.0, 0.1, 1.0
        )
        gui_handles["limit_weight"] = server.gui.add_slider(
            "Limit", 0.0, 100.0, 0.1, 100.0
        )
        gui_handles["rest_weight"] = server.gui.add_slider(
            "Rest", 0.0, 0.1, 0.001, 0.01
        )
        gui_handles["base_constraint_cost_weight"] = server.gui.add_slider(
            "Base Constr Wgt", 0.0, 1.0, 0.01, 0.1
        )

    gui_handles["add_joint_button"] = server.gui.add_button("Add joint")
    gui_handles["target_names"] = []
    gui_handles["target_tfs"] = []
    gui_handles["target_frames"] = []
    gui_handles["target_link_name_dropdowns"] = []

    def add_target_link_gui():
        idx = len(gui_handles["target_link_name_dropdowns"])
        name_handle = server.gui.add_dropdown(
            f"target link {idx}",
            list(robot.link.names),
            initial_value=robot.link.names[-1],
        )
        tf_handle = server.scene.add_transform_controls(
            f"target_transform_{idx}", scale=0.2
        )
        frame_handle = server.scene.add_frame(
            f"target_{idx}",
            axes_length=0.5 * 0.2,
            axes_radius=0.05 * 0.2,
            origin_radius=0.1 * 0.2,
        )
        gui_handles["target_link_name_dropdowns"].append(name_handle)
        gui_handles["target_tfs"].append(tf_handle)
        gui_handles["target_frames"].append(frame_handle)

    gui_handles["add_joint_button"].on_click(lambda _: add_target_link_gui())
    add_target_link_gui()

    return urdf_vis, gui_handles


def run_ik_loop(
    server: viser.ViserServer,
    robot: pk.Robot,
    urdf_vis: BatchedURDF,
    gui_handles: Dict[str, Any],
):
    """Runs the main IK solving and visualization loop for mobile base."""
    base_pose = jaxlie.SE3.identity()
    joints = (robot.joint.upper_limits_act + robot.joint.lower_limits_act) / 2

    while True:
        target_link_indices = jnp.array(
            [
                robot.link.names.index(h.value)
                for h in gui_handles["target_link_name_dropdowns"]
            ]
        )
        target_poses = jaxlie.SE3(
            jnp.stack(
                [jnp.array([*h.wxyz, *h.position]) for h in gui_handles["target_tfs"]]
            )
        )

        if gui_handles["smooth"].value:
            max_iter = 1
            init_joints = joints
            init_base_pose = base_pose
        else:
            max_iter = 100
            init_joints = None
            init_base_pose = None

        # Read base constraint checkboxes
        base_constraint_flags = jnp.array(
            [
                gui_handles["base_constraint_x"].value,
                gui_handles["base_constraint_y"].value,
                gui_handles["base_constraint_z"].value,
                gui_handles["base_constraint_rx"].value,
                gui_handles["base_constraint_ry"].value,
                gui_handles["base_constraint_rz"].value,
            ]
        )
        start_time = time.time()
        base_pose, joints = solve_ik(
            robot=robot,
            target_pose=target_poses,
            target_link_indices=target_link_indices,
            base_constraint_flags=base_constraint_flags,
            init_joints=init_joints,
            init_base_pose=init_base_pose,
            pos_weight=gui_handles["pos_weight"].value,
            rot_weight=gui_handles["rot_weight"].value,
            limit_weight=gui_handles["limit_weight"].value,
            rest_weight=gui_handles["rest_weight"].value,
            base_constraint_cost_weight=gui_handles[
                "base_constraint_cost_weight"
            ].value,
            max_iterations=max_iter,
        )
        jax.block_until_ready(joints)
        end_time = time.time()

        gui_handles["timing"].value = (end_time - start_time) * 1000

        # Update base pose visualization
        base_frame_handle = gui_handles["base_frame"]
        base_frame_handle.position = onp.array(base_pose.translation())
        base_frame_handle.wxyz = onp.array(base_pose.rotation().wxyz)

        # urdf_vis.update_base_pose(base_pose) # Original code might not have had this either
        urdf_vis.update_cfg(joints)

        # --- Original Logic for visualizing target frames ---
        # Get world poses of target frames for visualization
        Ts_link_base_array = robot.forward_kinematics(
            joints
        )  # Get link poses relative to base
        Ts_world_link_object = base_pose @ jaxlie.SE3(
            Ts_link_base_array
        )  # Transform to world as SE3 object

        for i, frame_handle in enumerate(gui_handles["target_frames"]):
            # Index into the numerical array of the SE3 object
            current_pose = jaxlie.SE3(
                Ts_world_link_object.wxyz_xyz[target_link_indices[i]]
            )
            frame_handle.position = onp.array(current_pose.translation().squeeze())
            frame_handle.wxyz = onp.array(current_pose.rotation().wxyz.squeeze())
        # --- End Original Logic ---


def main(
    device: Literal["cpu", "gpu"] = "cpu",
    robot_description: Optional[str] = "panda",
    robot_urdf_path: Optional[Path] = None,
):
    """Main function for mobile base IK example."""
    jax.config.update("jax_platform_name", device)
    logger.info(f"Using JAX device: {device}")

    urdf, robot = setup_robot(robot_description, robot_urdf_path)

    server = viser.ViserServer()
    urdf_vis, gui_handles = setup_visualization_and_gui(server, urdf, robot)

    run_ik_loop(
        server,
        robot,
        urdf_vis,
        gui_handles,
    )


if __name__ == "__main__":
    tyro.cli(main)
