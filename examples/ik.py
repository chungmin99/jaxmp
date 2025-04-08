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
from pyroki.viewer._batched_urdf import BatchedURDF


@jdc.jit
def solve_ik(
    robot: pk.Robot,
    target_pose: jaxlie.SE3,
    target_joint_indices: jnp.ndarray,
    init_joints: Optional[jnp.ndarray],
    *,
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    manipulability_weight: float | None = None,
    max_iterations: int = 1,
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
            weights=jnp.array([pos_weight] * 3 + [rot_weight] * 3),
        ),
        pk.LimitCost.make(
            robot,
            joint_var,
            weights=jnp.array([limit_weight] * robot.joint.count),
        ),
        pk.RestCost.make(
            joint_var,
            weights=jnp.array([rest_weight]),
        ),
    ]

    if manipulability_weight is not None:
        factors.append(
            pk.ManipulabilityCost.make(
                robot,
                joint_var,
                target_joint_indices,
                weights=jnp.array([manipulability_weight]),
            ),
        )

    if init_joints is not None:
        init_vars = [joint_var.with_value(init_joints)]
    else:
        init_vars = [joint_var]

    sol = pk.solve(vars, factors, init_vars=init_vars, max_iterations=max_iterations)
    return sol[joint_var]


def main(
    device: Literal["cpu", "gpu"] = "cpu",
    robot_description: Optional[str] = "panda",
    robot_urdf_path: Optional[Path] = None,
):
    # Set device.
    jax.config.update("jax_platform_name", device)

    # Load robot.
    urdf, robot = pk.load_robot(
        robot_urdf_path=robot_urdf_path, robot_description=robot_description
    )
    logger.info(
        "Loaded robot with {} joints, {} actuated joints, and {} links.",
        robot.joint.count,
        robot.joint.actuated_count,
        robot.link.count,
    )

    # Visualization code.
    server = viser.ViserServer()
    server.scene.configure_default_lights()
    urdf_vis = BatchedURDF(server, urdf, root_node_name="/base")
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    smooth_handle = server.gui.add_checkbox("DiffIK", initial_value=False)
    with server.gui.add_folder("Cost weights"):
        pos_weight_handle = server.gui.add_slider("Position", 0.0, 50.0, 0.1, 5.0)
        rot_weight_handle = server.gui.add_slider("Rotation", 0.0, 10.0, 0.1, 1.0)
        limit_weight_handle = server.gui.add_slider("Limit", 0.0, 100.0, 0.1, 100.0)
        manipulability_weight_handle = server.gui.add_slider(
            "Manipulability", 0.0, 0.01, 0.001, 0.0
        )
        rest_weight_handle = server.gui.add_slider("Rest", 0.0, 0.1, 0.001, 0.01)
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

    joints = (robot.joint.upper_limits_act + robot.joint.lower_limits_act) / 2
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

        if smooth_handle.value:
            max_iter = 1
            init_joints = joints
        else:
            max_iter = 100
            init_joints = None

        if manipulability_weight_handle.value > 0:
            manipulability_weight = manipulability_weight_handle.value
        else:
            manipulability_weight = None

        start = time.time()
        joints = solve_ik(
            robot,
            target_poses,
            target_joint_indices,
            init_joints=init_joints,
            pos_weight=pos_weight_handle.value,
            rot_weight=rot_weight_handle.value,
            limit_weight=limit_weight_handle.value,
            rest_weight=rest_weight_handle.value,
            manipulability_weight=manipulability_weight,
            max_iterations=max_iter,
        )
        jax.block_until_ready(joints)
        end = time.time()

        timing_handle.value = (end - start) * 1000
        urdf_vis.update_cfg(joints)

        Ts_joint_world = robot.forward_kinematics(joints)
        for i in range(len(target_name_handles)):
            pose = jaxlie.SE3(Ts_joint_world[target_joint_indices[i]])
            target_frame_handles[i].position = onp.array(pose.translation().squeeze())
            target_frame_handles[i].wxyz = onp.array(pose.rotation().wxyz.squeeze())


if __name__ == "__main__":
    tyro.cli(main)
