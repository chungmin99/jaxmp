from typing import Literal, Optional

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import jax_dataclasses as jdc

from jaxmp.robot_factors import RobotFactors
from jaxmp.kinematics import JaxKinTree
from jaxmp.coll import RobotColl, CollGeom


@jdc.jit
def solve_ik(
    kin: JaxKinTree,
    target_pose: jaxlie.SE3,
    target_joint_indices: jax.Array,
    rest_pose: jnp.ndarray,
    *,
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    manipulability_weight: float = 0.0,
    include_manipulability: jdc.Static[bool] = False,
    joint_vel_weight: float = 0.0,
    self_coll_weight: float = 2.0,
    world_coll_weight: float = 10.0,
    robot_coll: Optional[RobotColl] = None,
    world_coll_list: list[CollGeom] = [],
    solver_type: jdc.Static[
        Literal["cholmod", "conjugate_gradient", "dense_cholesky"]
    ] = "conjugate_gradient",
    freeze_target_xyz_xyz: Optional[jnp.ndarray] = None,
    freeze_base_xyz_xyz: Optional[jnp.ndarray] = None,
    dt: float = 0.01,
) -> tuple[jaxlie.SE3, jnp.ndarray]:
    """
    Solve IK for the robot.
    Args:
        target_pose: Desired pose of the target joint, SE3 has batch axes (n_target,).
        target_joint_indices: Indices of the target joints, length n_target.
        freeze_target_xyz_xyz: 6D vector indicating which axes to freeze in the target frame.
        freeze_base_xyz_xyz: 6D vector indicating which axes to freeze in the base frame.
    Returns:
        Base pose (jaxlie.SE3)
        Joint angles (jnp.ndarray)
    """
    if freeze_target_xyz_xyz is None:
        freeze_target_xyz_xyz = jnp.ones(6)
    if freeze_base_xyz_xyz is None:
        freeze_base_xyz_xyz = jnp.ones(6)

    JointVar = RobotFactors.get_var_class(kin, default_val=rest_pose)
    ConstrainedSE3Var = RobotFactors.get_constrained_se3(freeze_base_xyz_xyz)

    joint_vars = [JointVar(0), ConstrainedSE3Var(0)]

    factors: list[jaxls.Factor] = [
        jaxls.Factor(
            RobotFactors.limit_cost,
            (
                kin,
                JointVar(0),
                jnp.array([limit_weight] * kin.num_actuated_joints),
            ),
        ),
        jaxls.Factor(
            RobotFactors.joint_limit_vel_cost,
            (
                kin,
                JointVar(0),
                rest_pose,
                dt,
                jnp.array([joint_vel_weight] * kin.num_actuated_joints),
            ),
        ),
        jaxls.Factor(
            RobotFactors.rest_cost,
            (
                JointVar(0),
                jnp.array([rest_weight] * kin.num_actuated_joints),
            ),
        ),
    ]

    ik_weights = jnp.array([pos_weight] * 3 + [rot_weight] * 3)
    ik_weights = ik_weights * freeze_target_xyz_xyz
    factors.append(
        jaxls.Factor(
            RobotFactors.ik_cost,
            (
                kin,
                joint_vars[0],
                target_pose,
                target_joint_indices,
                ik_weights,
                ConstrainedSE3Var(0),
            ),
        ),
    )

    if include_manipulability:
        for idx, target_joint_idx in enumerate(target_joint_indices):
            factors.append(
                jaxls.Factor(
                    RobotFactors.manipulability_cost,
                    (
                        kin,
                        joint_vars[0],
                        target_joint_idx,
                        jnp.array([manipulability_weight] * kin.num_actuated_joints),
                    ),
                )
            )

    if robot_coll is not None:
        factors.append(
            jaxls.Factor(
                RobotFactors.self_coll_cost,
                (
                    kin,
                    robot_coll,
                    JointVar(0),
                    0.01,
                    jnp.full(robot_coll.coll.get_batch_axes(), self_coll_weight),
                    ConstrainedSE3Var(0),
                ),
            ),
        )
        for world_coll in world_coll_list:
            factors.append(
                jaxls.Factor(
                    RobotFactors.world_coll_cost,
                    (
                        kin,
                        robot_coll,
                        JointVar(0),
                        world_coll,
                        0.1,
                        jnp.full(robot_coll.coll.get_batch_axes(), world_coll_weight),
                        ConstrainedSE3Var(0),
                    ),
                ),
            )

    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars,
        use_onp=False,
    )
    solution = graph.solve(
        linear_solver=solver_type,
        initial_vals=jaxls.VarValues.make(joint_vars),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5,
            parameter_tolerance=1e-5,
            max_iterations=40,
        ),
        verbose=False,
    )

    # Update visualization.
    base_pose = solution[ConstrainedSE3Var(0)]
    joints = solution[JointVar(0)]
    return base_pose, joints
