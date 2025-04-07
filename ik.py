"""ik.py
Tests robot inverse kinematics using Pyroki.
"""

import time
from typing import Literal, Optional
from pathlib import Path
from loguru import logger
import tyro

import viser
import viser.extras

import jax
import jax.numpy as jnp
import jaxlie
import jax_dataclasses as jdc
import numpy as onp

import pyroki as pk


@jdc.jit
def solve_ik(
    robot: pk.Robot,
    target_pose: jaxlie.SE3,
    target_joint_indices: jnp.ndarray,
) -> jax.Array:
    """
    Solve the robot inverse kinematics problem, using PyRoki.
    Here, we minimize the pose and joint limit costs.
    """
    joint_var = robot.JointVar(0)
    vars = [joint_var]
    factors = [
        pk.PoseCost.make(
            robot,
            joint_var,
            target_pose,
            target_joint_indices,
            weights=jnp.array([5.0] * 3 + [1.0] * 3),
        ),
        pk.LimitCost.make(
            robot,
            joint_var,
            weights=jnp.array([100.0] * robot.joint.count),
        ),
    ]
    sol = pk.solve(vars, factors)
    return sol[joint_var]


def main(
    device: Literal["cpu", "gpu"] = "cpu",
    robot_description: Optional[str] = "panda",
    robot_urdf_path: Optional[Path] = None,
):
    # Set device.
    jax.config.update("jax_platform_name", device)

    # Load robot description.
    urdf, robot = pk.load_robot(
        robot_urdf_path=robot_urdf_path, robot_description=robot_description
    )
    logger.info(
        "Loaded robot with {} joints, {} actuated joints, and {} links.",
        robot.joint.count,
        robot.joint.actuated_count,
        robot.link.count,
    )

    server = viser.ViserServer()
    server.scene.configure_default_lights()
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    add_joint_button = server.gui.add_button("Add joint")

    target_name_handles: list[viser.GuiDropdownHandle] = []
    target_tf_handles: list[viser.TransformControlsHandle] = []
    target_frame_handles: list[viser.FrameHandle] = []

    def add_joint():
        idx = len(target_name_handles)
        target_name_handle = server.gui.add_dropdown(
            f"target joint {idx}",
            list(robot.joint.names),
            initial_value=robot.joint.names[0],
        )
        target_tf_handle = server.scene.add_transform_controls(
            f"target_transform_{idx}", scale=0.2
        )
        target_frame_handle = server.scene.add_frame(
            f"target_{idx}",
            axes_length=0.5 * 0.2,
            axes_radius=0.05 * 0.2,
            origin_radius=0.1 * 0.2,
        )
        target_name_handles.append(target_name_handle)
        target_tf_handles.append(target_tf_handle)
        target_frame_handles.append(target_frame_handle)

    add_joint_button.on_click(lambda _: add_joint())
    add_joint()

    while True:
        target_joint_indices = jnp.array(
            [
                robot.joint.names.index(target_name_handles[i].value)
                for i in range(len(target_name_handles))
            ]
        )
        target_poses = jaxlie.SE3(
            jnp.stack(
                [
                    jnp.array(
                        [*target_tf_handles[i].wxyz, *target_tf_handles[i].position]
                    )
                    for i in range(len(target_name_handles))
                ]
            )
        )

        start = time.time()
        joints = solve_ik(robot, target_poses, target_joint_indices)
        jax.block_until_ready(joints)
        end = time.time()

        timing_handle.value = (end - start) * 1000
        urdf_vis.update_cfg(onp.array(joints))

        Ts_joint_world = robot.forward_kinematics(joints)
        for i in range(len(target_name_handles)):
            pose = jaxlie.SE3(Ts_joint_world[target_joint_indices[i]])
            target_frame_handles[i].position = onp.array(pose.translation().squeeze())
            target_frame_handles[i].wxyz = onp.array(pose.rotation().wxyz.squeeze())


if __name__ == "__main__":
    tyro.cli(main)
