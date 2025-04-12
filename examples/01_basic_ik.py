"""01_basic_ik.py
Simplest Inverse Kinematics Example using PyRoKi.
"""

import time
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import pyroki as pk
import viser
from pyroki.viewer import BatchedURDF
from robot_descriptions.loaders.yourdfpy import load_robot_description


@jdc.jit
def solve_ik(
    robot: pk.Robot,
    T_world_target: jaxlie.SE3,
    target_joint_indices: jnp.ndarray,
) -> tuple[jax.Array, jax.Array]:
    """Solves the basic IK problem for a robot. Returns joint configuration."""
    joint_var = robot.JointVar(0)
    vars = [joint_var]
    factors = [
        pk.PoseCost(
            (
                joint_var,
                T_world_target,
            ),
            robot=robot,
            target_link_indices=target_joint_indices,
            weights=jnp.array([5.0] * 3 + [1.0] * 3),
        ),
        pk.LimitCost(
            (joint_var,),
            robot=robot,
            weights=jnp.array([100.0] * robot.joint.count),
        ),
        pk.RestCost(
            (joint_var,),
            rest_pose=joint_var.default_factory(),
            weights=jnp.array([0.001] * robot.joint.actuated_count),
        ),
    ]
    sol, cost = pk.solve(vars, factors, init_vars=[joint_var])
    return sol[joint_var], cost


def main():
    """Main function for basic IK (no collision or manipulability)."""

    urdf = load_robot_description("panda_description")
    ee_joint_idx = 9

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = BatchedURDF(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        # Solve IK.
        start_time = time.time()
        solution, cost = solve_ik(
            robot,
            T_world_target=jaxlie.SE3(
                jnp.concatenate([ik_target.wxyz, ik_target.position])
            ),
            target_joint_indices=jnp.array(ee_joint_idx),
        )
        jax.block_until_ready((solution, cost))
        
        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()
