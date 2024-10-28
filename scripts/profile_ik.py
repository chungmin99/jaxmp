"""
Profile IK speed.
"""

import time
from typing import Literal
import jax
import jax.numpy as jnp
import jaxlie
from loguru import logger

from jaxmp import JaxKinTree
from jaxmp.coll import RobotColl
from jaxmp.extras import load_urdf, solve_ik
import tyro

ROBOT_EE_JOINTS = {
    "panda": "panda_hand_tcp_joint",
    "ur5": "ee_fixed_joint",
    "yumi": "yumi_link_7_l_joint",
}


def profile_ik(
    robot_description: str,
    ee_joint_name: str,
    batch_size_list: list[int],
    device: Literal["cpu", "gpu", "default"],
    solver_type: Literal["conjugate_gradient", "dense_cholesky", "cholmod"],
    with_collision: bool,
    n_trials: int,
):
    if device != "default":
        jax.config.update("jax_platform_name", device)

    urdf = load_urdf(robot_description)
    kin = JaxKinTree.from_urdf(urdf)
    coll = RobotColl.from_urdf(urdf) if with_collision else None
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    random_key = jax.random.PRNGKey(0)

    joint_idx = jnp.array([kin.joint_names.index(ee_joint_name)])

    solve_ik_vmap = jax.vmap(
        lambda pose: solve_ik(
            kin,
            pose,
            joint_idx,
            rest_pose,
            solver_type=solver_type,
            robot_coll=coll,
        ),
    )

    for batch_size in batch_size_list:
        random_poses = jaxlie.SE3.sample_uniform(random_key, batch_axes=(batch_size,))

        # With JIT compile + run.
        start = time.time()
        _, joints = solve_ik_vmap(random_poses)
        jax.block_until_ready((_, joints))
        end = time.time()
        elapsed_with_jit = end - start

        # Without JIT compile.
        elapsed = 0
        for _ in range(n_trials):
            random_poses = jaxlie.SE3.sample_uniform(random_key, batch_axes=(batch_size,))
            start = time.time()
            _, joints = solve_ik_vmap(random_poses)
            jax.block_until_ready((_, joints))
            end = time.time()
            elapsed += end - start
        elapsed /= n_trials

        Ts_solved = kin.forward_kinematics(joints)[..., joint_idx, :]
        errors = jaxlie.SE3.log(jaxlie.SE3(Ts_solved) @ random_poses.inverse())

        logger.info(
            "For IK on {} with {} samples, with {} DoF{}:".format(
                device,
                batch_size,
                kin.num_actuated_joints,
                ", with collision" if with_collision else "",
            )
        )
        logger.info(
            f"\tElapsed time (seconds): {elapsed:10.6f} ({elapsed_with_jit:.6f} with JIT)"
        )
        logger.info(f"\tMean translation error: {errors[..., :3].mean():10.6f}")
        logger.info(f"\tMean rotation error:    {errors[..., 3:].mean():10.6f}")


def profile(
    robot: Literal["panda", "ur5", "yumi"] = "panda",
    device: Literal["cpu", "gpu", "default"] = "default",
    solver_type: Literal[
        "conjugate_gradient", "dense_cholesky", "cholmod"
    ] = "conjugate_gradient",
    with_collision: bool = False,
    n_trials: int = 5,
):
    # TODO this isn't representative speed-wise, because the samples 
    # are from a unit sphere and not from the reachable workspace.

    logger.disable("jaxmp")
    logger.disable("jaxls")

    ee_joint_name = ROBOT_EE_JOINTS[robot]
    batch_size_list = [1, 10, 100, 1000]

    profile_ik(
        robot_description=robot,
        ee_joint_name=ee_joint_name,
        batch_size_list=batch_size_list,
        device=device,  # type: ignore
        solver_type=solver_type,  # type: ignore
        with_collision=with_collision,
        n_trials=n_trials,
    )


if __name__ == "__main__":
    tyro.cli(profile)
