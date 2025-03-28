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
        "Loaded robot with {} joints and {} actuated joints.",
        robot.num_joints,
        robot.num_actuated_joints,
    )

    server = viser.ViserServer()
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")
    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(robot.joint_names),
        initial_value=robot.joint_names[0],
    )
    target_tf_handle = server.scene.add_transform_controls(
        "target_transform", scale=0.2
    )
    target_frame_handle = server.scene.add_frame(
        "target",
        axes_length=0.5 * 0.2,
        axes_radius=0.05 * 0.2,
        origin_radius=0.1 * 0.2,
    )
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    while True:
        target_joint_idx = robot.joint_names.index(target_name_handle.value)
        target_pose = jaxlie.SE3(
            jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position])[None]
        )

        start = time.time()
        joints = solve_ik(robot, target_pose, jnp.array([target_joint_idx]))
        jax.block_until_ready(joints)
        end = time.time()

        timing_handle.value = (end - start) * 1000
        urdf_vis.update_cfg(onp.array(joints))

        Ts_joint_world = robot.forward_kinematics(joints)
        pose = jaxlie.SE3(Ts_joint_world[target_joint_idx])
        target_frame_handle.position = onp.array(pose.translation().squeeze())
        target_frame_handle.wxyz = onp.array(pose.rotation().wxyz.squeeze())


@jdc.jit
def solve_ik(
    robot: pk.Robot,
    target_pose: jaxlie.SE3,
    target_joint_indices: jnp.ndarray,
) -> jax.Array:
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
            robot, joint_var, weights=jnp.array([100.0] * robot.num_actuated_joints)
        ),
    ]
    sol = pk.solve(vars, factors)
    return sol[joint_var]


if __name__ == "__main__":
    tyro.cli(main)
