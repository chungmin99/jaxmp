"""01_basic_ik.py
Simplest Inverse Kinematics Example using PyRoKi.

Demonstrates the core IK setup:
- Loading a robot model (URDF).
- Defining basic IK cost factors: Pose, Joint Limits, Rest Pose.
- Using the Pyroki solver (`pk.solve`).
- Visualizing the robot and target pose using Viser.
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
    init_joints: Optional[jnp.ndarray],
    *,
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    max_iterations: int = 1,
) -> jax.Array:
    """Solves the basic IK problem (no collision or manipulability).

    Args:
        robot: The Pyroki Robot model.
        target_pose: The desired SE(3) pose for the target link(s).
        target_link_indices: Indices of the links to target.
        init_joints: Initial guess for the joint configuration. If None, uses default.
        pos_weight: Weight for the position component of the pose cost.
        rot_weight: Weight for the rotation component of the pose cost.
        rest_weight: Weight for the cost penalizing deviation from the rest pose.
        limit_weight: Weight for the cost penalizing proximity to joint limits.
        max_iterations: Maximum number of iterations for the solver.

    Returns:
        The optimized joint configuration as a JAX array.
    """
    joint_var = robot.JointVar(0)
    vars = [joint_var]

    # Determine the rest pose: use init_joints if provided, else default
    if init_joints is not None:
        rest_pose_for_cost = init_joints
    else:
        rest_pose_for_cost = joint_var.default_factory()

    factors = [
        pk.PoseCost(
            (
                joint_var,
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
        pk.RestCost(
            (joint_var,),
            rest_pose=rest_pose_for_cost,
            weights=jnp.array([rest_weight] * robot.joint.actuated_count),
        ),
    ]

    if init_joints is not None:
        init_vars = [joint_var.with_value(init_joints)]
    else:
        init_vars = [joint_var]

    sol, _ = pk.solve(vars, factors, init_vars=init_vars, max_iterations=max_iterations)
    return sol[joint_var]


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
    logger.info("Setting up Viser server and GUI...")
    server.scene.configure_default_lights()

    urdf_vis = BatchedURDF(server, urdf, root_node_name="/base")
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    gui_handles = {}
    gui_handles["timing"] = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    gui_handles["smooth"] = server.gui.add_checkbox("DiffIK", initial_value=False)

    # Cost weight sliders.
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

    # Button and lists for multiple IK targets.
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
    """Runs the main basic IK solving and visualization loop."""
    joints = (robot.joint.upper_limits_act + robot.joint.lower_limits_act) / 2

    while True:
        target_link_indices = jnp.array(
            [robot.link.names.index(h.value) for h in gui_handles["target_link_name_dropdowns"]]
        )
        target_poses = jaxlie.SE3(
            jnp.stack(
                [jnp.array([*h.wxyz, *h.position]) for h in gui_handles["target_tfs"]]
            )
        )

        if gui_handles["smooth"].value:
            max_iter = 1
            init_joints = joints
        else:
            max_iter = 100
            init_joints = None

        start_time = time.time()
        joints = solve_ik(
            robot=robot,
            target_pose=target_poses,
            target_link_indices=target_link_indices,
            init_joints=init_joints,
            pos_weight=gui_handles["pos_weight"].value,
            rot_weight=gui_handles["rot_weight"].value,
            limit_weight=gui_handles["limit_weight"].value,
            rest_weight=gui_handles["rest_weight"].value,
            max_iterations=max_iter,
        )
        jax.block_until_ready(joints)
        end_time = time.time()

        gui_handles["timing"].value = (end_time - start_time) * 1000
        urdf_vis.update_cfg(joints)

        Ts_link_world = robot.forward_kinematics(joints)
        for i, frame_handle in enumerate(gui_handles["target_frames"]):
            current_pose = jaxlie.SE3(Ts_link_world[target_link_indices[i]])
            frame_handle.position = onp.array(current_pose.translation().squeeze())
            frame_handle.wxyz = onp.array(current_pose.rotation().wxyz.squeeze())


def main(
    device: Literal["cpu", "gpu"] = "cpu",
    robot_description: Optional[str] = "panda",
    robot_urdf_path: Optional[Path] = None,
):
    """Main function for basic IK (no collision or manipulability)."""
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
