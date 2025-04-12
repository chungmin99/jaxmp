"""02_ik_with_collision.py
Basic Inverse Kinematics with Collision Avoidance using PyRoKi.
"""

import time
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, Any, Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import tyro
import viser
from loguru import logger

import pyroki as pk
from pyroki.coll import CollGeom, HalfSpace, RobotCollision, Sphere
from pyroki.viewer._batched_urdf import BatchedURDF


@jdc.jit
def solve_ik(
    robot: pk.Robot,
    coll: RobotCollision,
    world_coll: Sequence[CollGeom],
    target_pose: jaxlie.SE3,
    target_link_indices: jnp.ndarray,
    init_joints: Optional[jnp.ndarray],
    *,
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    self_collision_weight: float = 2.0,
    world_collision_weight: float = 5.0,
    max_iterations: int = 1,
) -> jax.Array:
    """Solves the inverse kinematics problem with collision avoidance.

    Args:
        robot: The Pyroki Robot model.
        coll: The RobotCollision model for self-collision checking.
        world_coll: A list of collision geometries representing the environment.
        target_pose: The desired SE(3) pose for the target link(s).
        target_link_indices: Indices of the links to target.
        init_joints: Initial guess for the joint configuration. If None, uses the default configuration.
        pos_weight: Weight for the position component of the pose cost.
        rot_weight: Weight for the rotation component of the pose cost.
        rest_weight: Weight for the cost penalizing deviation from the rest pose.
        limit_weight: Weight for the cost penalizing proximity to joint limits.
        self_collision_weight: Weight for the self-collision avoidance cost.
        world_collision_weight: Weight for avoiding collisions with world objects.
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
        # Use direct constructor
        pk.PoseCost(
            (
                joint_var,
                target_pose,
            ),
            robot=robot,
            target_link_indices=target_link_indices,
            weights=jnp.array([pos_weight] * 3 + [rot_weight] * 3),
        ),
        # Use direct constructor
        pk.LimitCost(
            (joint_var,),
            robot=robot,
            weights=jnp.array([limit_weight] * robot.joint.count),
        ),
        # Use direct constructor
        pk.RestCost(
            (joint_var,), 
            rest_pose=rest_pose_for_cost, # Use the determined rest pose
            weights=jnp.array([rest_weight] * robot.joint.actuated_count),
        ),
    ]

    # Collision avoidance factors.
    # 1. Self-collision avoidance.
    factors.append(
        # Use direct constructor
        pk.SelfCollisionCost(
            (joint_var,),
            robot=robot,
            robot_coll=coll,
            margin=0.02,  # Collision distance threshold
            weights=jnp.array([self_collision_weight]),
        ),
    )
    # 2. World collision avoidance.
    for world_coll_geom in world_coll:
        factors.append(
            # Use direct constructor
            pk.WorldCollisionCost(
                (joint_var,),
                robot=robot,
                robot_coll=coll,
                world_geom=world_coll_geom,
                margin=0.05,  # Collision distance threshold
                weights=jnp.array([world_collision_weight]),
            )
        )

    if init_joints is not None:
        init_vars = [joint_var.with_value(init_joints)]
    else:
        init_vars = [joint_var]

    sol, _ = pk.solve(vars, factors, init_vars=init_vars, max_iterations=max_iterations)
    return sol[joint_var]


def setup_robot_and_collision(
    robot_description: Optional[str], robot_urdf_path: Optional[Path]
) -> Tuple[Any, pk.Robot, RobotCollision, HalfSpace, Sphere]:
    """Loads the robot model and sets up collision objects."""
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

    # Create robot self-collision model from URDF.
    coll = RobotCollision.from_urdf(urdf)

    # Define world collision geometries: plane (ground) and sphere (movable obstacle).
    plane_coll = HalfSpace.from_point_and_normal(
        jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        jnp.array([0.0, 0.0, 0.0]), jnp.array([0.05])
    )

    logger.info("Collision models created.")
    return urdf, robot, coll, plane_coll, sphere_coll


def setup_visualization_and_gui(
    server: viser.ViserServer,
    urdf: Any,
    robot: pk.Robot,
    sphere_coll: Sphere,
) -> Tuple[BatchedURDF, Dict[str, Any]]:
    """Initializes the Viser visualizer and GUI elements."""
    server.scene.configure_default_lights()

    # Add robot model and grid.
    urdf_vis = BatchedURDF(server, urdf, root_node_name="/base")
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    # --- GUI Elements ---
    gui_handles = {}
    gui_handles["timing"] = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    gui_handles["smooth"] = server.gui.add_checkbox("DiffIK", initial_value=False)

    # Movable sphere obstacle.
    gui_handles["sphere_coll"] = server.scene.add_transform_controls(
        "/sphere", scale=0.2
    )
    initial_sphere_pos_tuple = (0.5, 0.0, 0.3)
    gui_handles["sphere_coll"].position = initial_sphere_pos_tuple
    server.scene.add_mesh_trimesh("/sphere/mesh", mesh=sphere_coll.to_trimesh())

    # Visualization toggles folder
    with server.gui.add_folder("Visualization"):
        gui_handles["visualize_coll"] = server.gui.add_checkbox("Show Collbody", False)

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
        gui_handles["self_collision_weight"] = server.gui.add_slider(
            "Self collision", 0.0, 10.0, 0.1, 0.5
        )
        gui_handles["world_collision_weight"] = server.gui.add_slider(
            "World collision", 0.0, 10.0, 0.1, 1.0
        )

    # Button and lists for multiple IK targets.
    gui_handles["add_joint_button"] = server.gui.add_button("Add joint")
    gui_handles["target_names"] = []
    gui_handles["target_tfs"] = []
    gui_handles["target_frames"] = []
    gui_handles["target_link_name_dropdowns"] = []

    def add_target_link_gui():
        """Adds GUI elements for a new IK target."""
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
        gui_handles["target_names"].append(name_handle)
        gui_handles["target_tfs"].append(tf_handle)
        gui_handles["target_frames"].append(frame_handle)

    gui_handles["add_joint_button"].on_click(lambda _: add_target_link_gui())
    add_target_link_gui()  # Add the first target initially.

    return urdf_vis, gui_handles


def run_ik_loop(
    server: viser.ViserServer,
    robot: pk.Robot,
    coll: RobotCollision,
    plane_coll: HalfSpace,
    sphere_coll: Sphere,  # Add sphere_coll back
    urdf_vis: BatchedURDF,
    gui_handles: Dict[str, Any],
):
    """Runs the IK solving and visualization loop."""
    joints = (robot.joint.upper_limits_act + robot.joint.lower_limits_act) / 2

    while True:
        # Get target link indices and poses from GUI.
        target_link_indices = jnp.array(
            [robot.link.names.index(h.value) for h in gui_handles["target_names"]],
            dtype=jnp.int32,  # Ensure integer type for indexing
        )
        target_poses = jaxlie.SE3(
            jnp.stack(
                [jnp.array([*h.wxyz, *h.position]) for h in gui_handles["target_tfs"]]
            )
        )

        # Determine solver settings (smooth vs full optimization).
        if gui_handles["smooth"].value:
            max_iter = 1
            init_joints = joints
        else:
            max_iter = 100
            init_joints = None

        # Update sphere obstacle pose based on GUI handle.
        sphere_tf_handle = gui_handles["sphere_coll"]
        T_sphere_world = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.array(sphere_tf_handle.wxyz)),
            jnp.array(sphere_tf_handle.position),
        )
        sphere_coll_world = sphere_coll.transform(T_sphere_world)

        # Combine static plane and dynamic sphere.
        world_coll: Sequence[CollGeom] = [plane_coll, sphere_coll_world]

        # Solve IK.
        start_time = time.time()
        joints = solve_ik(
            robot=robot,
            coll=coll,
            world_coll=world_coll, # Use updated world_coll
            target_pose=target_poses,
            target_link_indices=target_link_indices, # Pass link indices
            init_joints=init_joints,
            pos_weight=gui_handles["pos_weight"].value,
            rot_weight=gui_handles["rot_weight"].value,
            limit_weight=gui_handles["limit_weight"].value,
            rest_weight=gui_handles["rest_weight"].value,
            self_collision_weight=gui_handles["self_collision_weight"].value,
            world_collision_weight=gui_handles["world_collision_weight"].value,
            max_iterations=max_iter,
        )
        jax.block_until_ready(joints)
        end_time = time.time()

        # Update GUI and visualization.
        gui_handles["timing"].value = (end_time - start_time) * 1000
        urdf_vis.update_cfg(joints)

        Ts_link_world_array = robot.forward_kinematics(joints)
        for i, frame_handle in enumerate(gui_handles["target_frames"]):
            # Index the numerical array to get the specific link's world pose array
            current_pose_array = Ts_link_world_array[target_link_indices[i]]
            # Convert back to SE3 object to access methods
            current_pose = jaxlie.SE3(current_pose_array)
            frame_handle.position = onp.array(current_pose.translation().squeeze())
            frame_handle.wxyz = onp.array(current_pose.rotation().wxyz.squeeze())


def main(
    device: Literal["cpu", "gpu"] = "cpu",
    robot_description: Optional[str] = "panda",
    robot_urdf_path: Optional[Path] = None,
):
    """Main function to run the basic IK with collision example."""
    jax.config.update("jax_platform_name", device)
    logger.info(f"Using JAX device: {device}")

    urdf, robot, coll, plane_coll, sphere_coll = setup_robot_and_collision(
        robot_description, robot_urdf_path
    )

    server = viser.ViserServer()
    urdf_vis, gui_handles = setup_visualization_and_gui(
        server, urdf, robot, sphere_coll
    )

    run_ik_loop(
        server,
        robot,
        coll,
        plane_coll,
        sphere_coll,
        urdf_vis,
        gui_handles,
    )


if __name__ == "__main__":
    tyro.cli(main)
