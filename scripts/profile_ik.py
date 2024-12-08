"""
Profile IK speed.
"""

import time
from typing import Literal
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from loguru import logger

import jaxls
from jaxmp import JaxKinTree, RobotFactors
from jaxmp.coll import RobotColl
from jaxmp.extras import load_urdf
import tyro

ROBOT_EE_JOINTS = {
    "panda": "panda_hand_tcp_joint",
    "ur5": "ee_fixed_joint",
    "yumi": "yumi_link_7_l_joint",
}


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
    coll = RobotColl.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    random_key = jax.random.PRNGKey(0)

    joint_idx = jnp.array([kin.joint_names.index(ee_joint_name)])
    JointVar = RobotFactors.get_var_class(kin, rest_pose)

    solve_ik_vmap = jax.vmap(
        lambda pose: solve_ik(
            kin,
            pose,
            joint_idx,
            rest_pose,
            JointVar,
            coll,
            include_self_coll=with_collision,
            solver_type=solver_type,
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
            random_poses = jaxlie.SE3.sample_uniform(
                random_key, batch_axes=(batch_size,)
            )
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


@jdc.jit
def solve_ik(
    kin: JaxKinTree,
    target_pose: jaxlie.SE3,
    target_joint_indices: jax.Array,
    initial_pose: jnp.ndarray,
    JointVar: jdc.Static[type[jaxls.Var[jax.Array]]],
    coll: RobotColl,
    include_self_coll: jdc.Static[bool] = False,
    *,
    joint_var_idx: int = 0,
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    joint_vel_weight: float = 0.0,
    self_coll_weight: float = 5.0,
    dt: float = 0.01,
    solver_type: jdc.Static[
        Literal["cholmod", "conjugate_gradient", "dense_cholesky"]
    ] = "conjugate_gradient",
    ConstrainedSE3Var: jdc.Static[type[jaxls.Var[jaxlie.SE3]] | None] = None,
    pose_var_idx: int = 0,
    max_iterations: int = 50,
) -> tuple[jaxlie.SE3, jnp.ndarray]:
    """
    Solve IK for the robot.
    Args:
        target_pose: Desired pose of the target joint, SE3 has batch axes (n_target,).
        target_joint_indices: Indices of the target joints, length n_target.
        initial_pose: Initial pose of the joints, used for joint velocity cost factor.
        JointVar: Joint variable type.
        ConstrainedSE3Var: Constrained SE3 variable type.
        joint_var_idx: Index for the joint variable.
        pose_var_idx: Index for the pose variable.
        ik_weight: Weight for the IK cost factor.
        rest_weight: Weight for the rest cost factor.
        limit_weight: Weight for the joint limit cost factor.
        joint_vel_weight: Weight for the joint velocity cost factor.
        solver_type: Type of solver to use.
        dt: Time step for the velocity cost factor.
        max_iterations: Maximum number of iterations for the solver.
    Returns:
        Base pose (jaxlie.SE3)
        Joint angles (jnp.ndarray)
    """
    # NOTE You can't add new factors on-the-fly with JIT, because:
    # - we'd want to pass in lists of jaxls.Factor objects
    # - but lists / tuples are static
    # - and ArrayImpl is not a valid type for a static argument.
    # (and you can't stack different Factor definitions, since it's a part of the treedef.)

    ik_weight = jnp.array([pos_weight] * 3 + [rot_weight] * 3)
    factors: list[jaxls.Factor] = [
        RobotFactors.limit_cost_factor(
            JointVar,
            joint_var_idx,
            kin,
            jnp.array([limit_weight] * kin.num_actuated_joints),
        ),
        RobotFactors.limit_vel_cost_factor(
            JointVar,
            joint_var_idx,
            kin,
            dt,
            jnp.array([joint_vel_weight] * kin.num_actuated_joints),
            prev_cfg=initial_pose,
        ),
        RobotFactors.rest_cost_factor(
            JointVar,
            joint_var_idx,
            jnp.array([rest_weight] * kin.num_actuated_joints),
        ),
    ]

    factors.append(
        RobotFactors.ik_cost_factor(
            JointVar,
            joint_var_idx,
            kin,
            target_pose,
            target_joint_indices,
            ik_weight,
            BaseConstrainedSE3VarType=ConstrainedSE3Var,
            base_se3_var_idx=pose_var_idx,
        ),
    )

    joint_vars: list[jaxls.Var] = [JointVar(joint_var_idx)]
    joint_var_values: list[jaxls.Var | jaxls._variables.VarWithValue] = [
        JointVar(joint_var_idx).with_value(initial_pose)
    ]
    if ConstrainedSE3Var is not None and pose_var_idx is not None:
        joint_vars.append(ConstrainedSE3Var(pose_var_idx))
        joint_var_values.append(ConstrainedSE3Var(pose_var_idx))

    if include_self_coll:
        factors.append(
            RobotFactors.self_coll_factor(
                JointVar, joint_var_idx, kin, coll, 0.05, self_coll_weight
            )
        )

    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars,
        use_onp=False,
    )
    solution = graph.solve(
        linear_solver=solver_type,
        initial_vals=jaxls.VarValues.make(joint_var_values),
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5,
            parameter_tolerance=1e-5,
            max_iterations=max_iterations,
        ),
        verbose=False,
    )

    if ConstrainedSE3Var is not None:
        base_pose = solution[ConstrainedSE3Var(0)]
    else:
        base_pose = jaxlie.SE3.identity()

    joints = solution[JointVar(0)]
    return base_pose, joints


if __name__ == "__main__":
    tyro.cli(profile)
