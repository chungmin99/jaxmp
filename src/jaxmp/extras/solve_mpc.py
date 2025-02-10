from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import jax_dataclasses as jdc

from jaxmp.robot_factors import RobotFactors
from jaxmp.kinematics import JaxKinTree
from jaxmp.coll import RobotColl, CollGeom


@jdc.jit
def solve_mpc(
    kin: JaxKinTree,
    robot_coll: RobotColl,
    world_coll_list: list[CollGeom],
    target_pose: jaxlie.SE3,
    target_joint_indices: jax.Array,
    initial_joints: jnp.ndarray,
    JointVar: jdc.Static[type[jaxls.Var[jax.Array]]],
    n_steps: jdc.Static[int],
    dt: float = 0.2,
    *,
    pos_weight: float = 5.0,
    rot_weight: float = 2.0,
    limit_weight: float = 100.0,
    joint_vel_weight: float = 10.0,
    joint_smoothness_weight: float = 10.0,
    pose_smoothness_weight: float = 1.0,
    use_manipulability: jdc.Static[bool] = False,
    manipulability_weight: float = 0.001,
    use_self_collision: jdc.Static[bool] = False,
    self_collision_weight: float | jax.Array = 20.0,
    use_world_collision: jdc.Static[bool] = False,
    world_collision_weight: float | jax.Array = 20.0,
    max_iterations: int = 50,
    prev_sols: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Plan next `n_steps` steps using MPC for the robot.
    Formulate MPC problem as a series of x and u optimization variables.
    - x: joint angles
    - u: joint velocities

    And returns the optimal next joint angles, and the associated cost.

    Goal is to:
    - Minimize distance to goal pose, at each timestep, and
    - Ensure that all constraints are satisfied:
        - Joint limits
        - Velocity limits
        - Collision avoidance
    """
    factors: list[jaxls.Factor] = []

    num_targets = target_joint_indices.shape[0]

    # Custom SE3 variable to batch across multiple joint targets.
    # This is not to be confused with SE3Vars with ids, which we use here for timesteps.
    class BatchedSE3Var(  # pylint: disable=missing-class-docstring
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.identity((num_targets,)),
        retract_fn=batched_rplus,
        tangent_dim=jaxlie.SE3.tangent_dim * num_targets,
    ): ...

    # We need to define a custom rplus operation that can batch across multiple
    # joint targets, while keeping SE3Var's`tangent_dim` property the same.
    # We could have created SE3Vars with ids=(tstep * num_joints + joint_idx),
    # but this feels more readable.
    def batched_rplus(
        pose: jaxlie.SE3,
        delta: jax.Array,
    ) -> jaxlie.SE3:
        return jax.vmap(jaxlie.manifold.rplus)(pose, delta.reshape(num_targets, -1))

    def match_joint_to_pose_cost(vals, joint_var, pose_var):
        joint_cfg = vals[joint_var]
        target_pose = vals[pose_var]
        Ts_joint_world = kin.forward_kinematics(joint_cfg)
        residual = (
            (jaxlie.SE3(Ts_joint_world[..., target_joint_indices, :])).inverse()
            @ (target_pose)
        ).log()
        return (residual * 100).flatten()

    # Define initial costs; match initial pose.
    match_initial_pose_cost = limit_weight
    start_pose = jaxlie.SE3(
        kin.forward_kinematics(initial_joints)[..., target_joint_indices, :]
    )
    factors.append(
        jaxls.Factor(
            lambda vals, val: (vals[val] - initial_joints) * match_initial_pose_cost,
            (JointVar(0),),
        )
    )

    # Define stage costs.
    factors.extend(
        [
            jaxls.Factor(
                match_joint_to_pose_cost,
                (
                    JointVar(jnp.arange(n_steps + 1)),
                    BatchedSE3Var(jnp.arange(n_steps + 1)),
                ),
            ),
            RobotFactors.limit_cost_factor(
                JointVar,
                jnp.arange(n_steps + 1),
                kin,
                jnp.array([limit_weight] * kin.num_actuated_joints),
            ),
        ]
    )

    # Smoothness / velocity-related costs.
    factors.extend(
        [
            jaxls.Factor(
                lambda vals, var_prev, var_curr: (
                    (vals[var_prev].inverse() @ vals[var_curr]).log().flatten()
                )
                * pose_smoothness_weight,
                (
                    BatchedSE3Var(jnp.arange(0, n_steps)),
                    BatchedSE3Var(jnp.arange(1, n_steps + 1)),
                ),
            ),
            jaxls.Factor(
                lambda vals, var_prev, var_curr: (
                    (vals[var_prev] - vals[var_curr]).flatten()
                )
                * joint_smoothness_weight,
                (
                    JointVar(jnp.arange(0, n_steps)),
                    JointVar(jnp.arange(1, n_steps + 1)),
                ),
            ),
            RobotFactors.limit_vel_cost_factor(
                JointVar,
                jnp.arange(1, n_steps + 1),
                kin,
                dt,
                jnp.array([joint_vel_weight] * kin.num_actuated_joints),
                prev_var_idx=jnp.arange(0, n_steps),
            ),
        ]
    )

    if use_manipulability:
        factors.append(
            RobotFactors.manipulability_cost_factor(
                JointVar,
                jnp.arange(n_steps + 1),
                kin,
                target_joint_indices,
                manipulability_weight,
            )
        )

    if use_self_collision:
        factors.append(
            RobotFactors.self_coll_factor(
                JointVar,
                jnp.arange(1, n_steps + 1),
                kin,
                robot_coll,
                activation_dist=0.05,
                weights=self_collision_weight,
                prev_var_idx=jnp.arange(0, n_steps),
            )
        )

    if use_world_collision:
        for world_coll in world_coll_list:
            factors.append(
                RobotFactors.world_coll_factor(
                    JointVar,
                    jnp.arange(1, n_steps + 1),
                    kin,
                    robot_coll,
                    world_coll,
                    activation_dist=0.10,
                    weights=world_collision_weight,
                    prev_var_idx=jnp.arange(0, n_steps),
                )
            )

    # Define terminal costs.
    factors.append(
        jaxls.Factor(
            lambda vals, var: (
                (vals[var].inverse() @ target_pose).log()
                * jnp.array([pos_weight] * 3 + [rot_weight] * 3)
            ).flatten()
            * n_steps,
            (BatchedSE3Var(n_steps),),
        )
    )

    joint_vars = [JointVar(idx) for idx in range(n_steps + 1)]
    pose_vars = [BatchedSE3Var(idx) for idx in range(n_steps + 1)]

    # Initialize variables using prev_sols if available.
    if prev_sols is None:
        prev_sols = initial_joints[None].repeat(n_steps, axis=0)
    else:
        assert prev_sols.shape[0] == n_steps

    joint_var_values = [
        JointVar(idx).with_value(prev_sols[idx]) for idx in range(n_steps)
    ]
    joint_var_values.append(JointVar(n_steps).with_value(prev_sols[-1]))

    pose_var_values = [pv.with_value(start_pose) for pv in pose_vars[:1]] + [
        pv.with_value(target_pose) for pv in pose_vars[1:]
    ]

    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars + pose_vars,
        use_onp=False,
    )
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(joint_var_values + pose_var_values),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(max_iterations=max_iterations),
        verbose=False,
    )

    joints = jnp.stack([solution[JointVar(idx)] for idx in range(1, n_steps + 1)])
    cost = jnp.sum(graph.compute_residual_vector(solution) ** 2)
    return joints, cost
