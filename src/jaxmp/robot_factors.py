"""Common `jaxls` factors for robot control, via wrapping `JaxKinTree` and `RobotColl`."""

from typing import Optional

import jax
from jax import Array
import jax.numpy as jnp
import jaxlie

import jaxls

from jaxmp.kinematics import JaxKinTree
from jaxmp.coll import CollGeom, collide, colldist_from_sdf, RobotColl


class RobotFactors:
    """Helper class for using `jaxls` factors with a `JaxKinTree` and `RobotColl`."""

    @staticmethod
    def get_var_class(
        kin: JaxKinTree, default_val: Optional[Array] = None
    ) -> type[jaxls.Var[Array]]:
        """Get the Variable class for this robot. Default value is `(limits_upper + limits_lower) / 2`."""
        if default_val is None:
            default_val = (kin.limits_upper + kin.limits_lower) / 2

        class JointVar(  # pylint: disable=missing-class-docstring
            jaxls.Var[Array],
            default_factory=lambda: default_val.copy(),
            tangent_dim=kin.num_actuated_joints,
            retract_fn=kin.get_retract_fn(),
        ): ...

        return JointVar

    @staticmethod
    def get_constrained_se3(
        freeze_base_xyz_xyz: Array,
    ) -> type[jaxls.Var[jaxlie.SE3]]:
        """Create a `SE3Var` with certain axes frozen.
        For all axes where it is 1, the corresponding update is ignored.
        """

        def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
            """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
            delta = delta * (1 - freeze_base_xyz_xyz)
            return jaxls.SE3Var.retract_fn(transform, delta)

        class ConstrainedSE3Var(
            jaxls.Var[jaxlie.SE3],
            default_factory=lambda: jaxlie.SE3.identity(),
            tangent_dim=jaxlie.SE3.tangent_dim,
            retract_fn=retract_fn,
        ): ...

        return ConstrainedSE3Var

    @staticmethod
    def ik_cost_factor(
        JointVarType: type[jaxls.Var[Array]],
        var_idx: jax.Array | int,
        kin: JaxKinTree,
        target_pose: jaxlie.SE3,
        target_joint_idx: jax.Array,
        weights: Array,
        BaseConstrainedSE3VarType: Optional[type[jaxls.Var[jaxlie.SE3]]] = None,
        base_se3_var_idx: Optional[jax.Array | int] = None,
        base_se3: Optional[jaxlie.SE3] = None,
        OffsetConstrainedSE3VarType: Optional[type[jaxls.Var[jaxlie.SE3]]] = None,
        offset_se3_var_idx: Optional[jax.Array | int] = None,
        offset_se3: Optional[jaxlie.SE3] = None,
    ) -> jaxls.Factor:
        """Pose cost."""

        def ik_cost(
            vals: jaxls.VarValues,
            kin: JaxKinTree,
            target_pose: jaxlie.SE3,
            target_joint_idx: jax.Array,
            var: jaxls.Var[Array],
            weights: Array,
            base_tf_var: jaxls.Var[jaxlie.SE3] | jaxlie.SE3 | None = None,
            ee_tf_var: jaxls.Var[jaxlie.SE3] | jaxlie.SE3 | None = None,
        ):
            # Handle world-to-base transform.
            joint_cfg: jax.Array = vals[var]
            if isinstance(base_tf_var, jaxls.Var):
                base_tf = vals[base_tf_var]
            elif isinstance(base_tf_var, jaxlie.SE3):
                base_tf = base_tf_var
            else:
                base_tf = jaxlie.SE3.identity()

            # Handle offset-to-target transform.
            if isinstance(ee_tf_var, jaxls.Var):
                ee_tf = vals[ee_tf_var]
            elif isinstance(ee_tf_var, jaxlie.SE3):
                ee_tf = ee_tf_var
            else:
                ee_tf = jaxlie.SE3.identity()

            Ts_joint_world = kin.forward_kinematics(joint_cfg)
            residual = (
                (
                    base_tf @ jaxlie.SE3(Ts_joint_world[target_joint_idx]) @ ee_tf
                ).inverse()
                @ (target_pose)
            ).log()
            weights = jnp.broadcast_to(weights, residual.shape)
            assert residual.shape == weights.shape
            return (residual * weights).flatten()

        # Handle optional base and offset transforms.
        # They may either be variables (optimizable) or fixed values.
        if BaseConstrainedSE3VarType is None or base_se3_var_idx is None:
            base_se3_var = None
        elif base_se3 is not None:
            base_se3_var = base_se3
        else:
            base_se3_var = BaseConstrainedSE3VarType(base_se3_var_idx)

        if OffsetConstrainedSE3VarType is None or offset_se3_var_idx is None:
            ee_se3_var = None
        elif offset_se3 is not None:
            ee_se3_var = offset_se3
        else:
            ee_se3_var = OffsetConstrainedSE3VarType(offset_se3_var_idx)

        return jaxls.Factor(
            ik_cost,
            (
                kin,
                target_pose,
                target_joint_idx,
                JointVarType(var_idx),
                weights,
                base_se3_var,
                ee_se3_var,
            ),
        )

    @staticmethod
    def limit_cost_factor(
        JointVarType: type[jaxls.Var[Array]],
        var_idx: jax.Array | int,
        kin: JaxKinTree,
        weights: Array,
    ) -> jaxls.Factor:
        """Limit cost."""

        def limit_cost(
            vals: jaxls.VarValues,
            kin: JaxKinTree,
            var: jaxls.Var[Array],
            weights: Array,
        ) -> Array:
            joint_cfg: jax.Array = vals[var]
            residual_upper = jnp.maximum(0.0, joint_cfg - kin.limits_upper)
            residual_lower = jnp.maximum(0.0, kin.limits_lower - joint_cfg)
            residual = residual_upper + residual_lower
            assert residual.shape == weights.shape
            return residual * weights

        return jaxls.Factor(limit_cost, (kin, JointVarType(var_idx), weights))

    @staticmethod
    def limit_vel_cost_factor(
        JointVarType: type[jaxls.Var[Array]],
        var_idx: jax.Array | int,
        kin: JaxKinTree,
        dt: float,
        weights: Array,
        prev_cfg: Optional[Array] = None,
        prev_var_idx: Optional[jax.Array] = None,
    ) -> jaxls.Factor:
        """Joint limit velocity cost."""

        def vel_limit_cost(
            vals: jaxls.VarValues,
            kin: JaxKinTree,
            var_curr: jaxls.Var[Array],
            var_prev: jaxls.Var[Array] | Array,
            dt: float,
            weights: Array,
        ) -> Array:
            prev = vals[var_prev] if isinstance(var_prev, jaxls.Var) else var_prev
            joint_vel = (vals[var_curr] - prev) / dt
            residual = jnp.maximum(0.0, jnp.abs(joint_vel) - kin.joint_vel_limit)
            assert residual.shape == weights.shape
            return residual * weights

        if prev_var_idx is not None:
            var_prev = JointVarType(prev_var_idx)
        elif prev_cfg is not None:
            var_prev = prev_cfg
        else:
            raise ValueError("Either prev_cfg or prev_var_idx must be provided.")

        return jaxls.Factor(
            vel_limit_cost,
            (kin, JointVarType(var_idx), var_prev, dt, weights),
        )

    @staticmethod
    def rest_cost_factor(
        JointVarType: type[jaxls.Var[Array]],
        var_idx: jax.Array | int,
        weights: Array,
    ) -> jaxls.Factor:
        """Bias towards joints at rest pose, specified by `default`."""

        def rest_cost(
            vals: jaxls.VarValues,
            var: jaxls.Var[Array],
            weights: Array,
        ) -> Array:
            default = var.default_factory()
            assert default is not None
            assert default.shape == vals[var].shape and default.shape == weights.shape
            return (vals[var] - default) * weights

        return jaxls.Factor(rest_cost, (JointVarType(var_idx), weights))

    @staticmethod
    def self_coll_factor(
        JointVarType: type[jaxls.Var[Array]],
        var_idx: jax.Array | int,
        kin: JaxKinTree,
        robot_coll: RobotColl,
        activation_dist: jax.Array | float,
        weights: jax.Array | float,
    ) -> jaxls.Factor:
        """Collision-scaled dist for self-collision.
        `activation_dist` and `weights` should be given in terms of the collision link pairs,
        e.g., through `RobotColl.coll_weight`.
        """

        # jaxls does not support static/varying treedefs, for batching.
        def self_coll_cost(
            vals: jaxls.VarValues,
            var: jaxls.Var[Array],
            activation_dist: jax.Array,
            weights: jax.Array,
            indices_0: jax.Array,
            indices_1: jax.Array,
        ) -> Array:
            joint_cfg = vals[var]
            colls = robot_coll.at_joints(kin, joint_cfg)
            assert isinstance(colls, CollGeom)
            coll_0 = colls.slice(..., indices_0)
            coll_1 = colls.slice(..., indices_1)

            sdf = collide(coll_0, coll_1).dist
            return (
                colldist_from_sdf(sdf, activation_dist=activation_dist) * weights
            ).flatten()

        assert isinstance(robot_coll.coll, CollGeom)

        if isinstance(weights, float) or isinstance(weights, int):
            weights = jnp.full((len(robot_coll.self_coll_list),), weights)
        else:
            assert len(weights) == len(robot_coll.self_coll_list)

        if isinstance(activation_dist, float) or isinstance(activation_dist, int):
            activation_dist = jnp.full(
                (len(robot_coll.self_coll_list),), activation_dist
            )
        else:
            assert len(activation_dist) == len(robot_coll.self_coll_list)

        num_coll_factors = len(robot_coll.self_coll_list)
        factor = jaxls.Factor(
            self_coll_cost,
            (
                JointVarType(jnp.array([var_idx] * num_coll_factors)),
                activation_dist,
                weights,
                jnp.asarray(
                    [
                        robot_coll.link_to_colls[pair[0]]
                        for pair in robot_coll.self_coll_list
                    ]
                ),
                jnp.asarray(
                    [
                        robot_coll.link_to_colls[pair[1]]
                        for pair in robot_coll.self_coll_list
                    ]
                ),
            ),
        )
        return factor

    @staticmethod
    def world_coll_factor(
        JointVarType: type[jaxls.Var[Array]],
        joint_idx: jax.Array | int,
        kin: JaxKinTree,
        robot_coll: RobotColl,
        other: CollGeom,
        activation_dist: float | jax.Array,
        weights: float | jax.Array,
        base_tf_var: jaxls.Var[jaxlie.SE3] | jaxlie.SE3 | None = None,
    ) -> jaxls.Factor:
        """Collision-scaled dist for world collision."""

        def world_coll_cost(
            vals: jaxls.VarValues,
            var: jaxls.Var[Array],
            eta: jax.Array,
            weights: jax.Array,
            coll_indices: jax.Array,
        ) -> Array:
            joint_cfg = vals[var]
            if isinstance(base_tf_var, jaxls.Var):
                base_tf = vals[base_tf_var]
            elif isinstance(base_tf_var, jaxlie.SE3):
                base_tf = base_tf_var
            else:
                base_tf = jaxlie.SE3.identity()

            coll = robot_coll.at_joints(kin, joint_cfg)
            assert isinstance(coll, CollGeom)
            coll = coll.slice(..., coll_indices).transform(base_tf)

            sdf = collide(coll, other).dist
            return (colldist_from_sdf(sdf, activation_dist=eta) * weights).flatten()

        assert isinstance(robot_coll.coll, CollGeom)

        if isinstance(weights, float) or isinstance(weights, int):
            weights = jnp.full((len(robot_coll.coll_link_names),), weights)
        else:
            assert len(weights) == len(robot_coll.coll_link_names)

        if isinstance(activation_dist, float) or isinstance(activation_dist, int):
            activation_dist = jnp.full(
                (len(robot_coll.coll_link_names),), activation_dist
            )
        else:
            assert len(activation_dist) == len(robot_coll.coll_link_names)

        num_coll_factors = len(robot_coll.coll_link_names)
        return jaxls.Factor(
            world_coll_cost,
            (
                JointVarType(jnp.array([joint_idx] * num_coll_factors)),
                activation_dist,
                weights,
                jnp.asarray(
                    [
                        robot_coll.link_to_colls[coll_idx]
                        for coll_idx in range(num_coll_factors)
                    ]
                ),
            ),
        )

    @staticmethod
    def smoothness_cost_factor(
        JointVarType: type[jaxls.Var[Array]],
        var_idx_curr: jax.Array | int,
        var_idx_past: jax.Array | int,
        weights: Array,
    ) -> jaxls.Factor:
        """Smoothness cost, for trajectories etc."""

        def smoothness_cost(
            vals: jaxls.VarValues,
            var_curr: jaxls.Var[Array],
            var_past: jaxls.Var[Array],
            weights: Array,
        ) -> Array:
            residual = vals[var_curr] - vals[var_past]
            assert residual.shape == weights.shape
            return residual * weights

        return jaxls.Factor(
            smoothness_cost,
            (
                JointVarType(var_idx_curr),
                JointVarType(var_idx_past),
                weights,
            ),
        )

    @staticmethod
    def manipulability_cost_factors(
        JointVarType: type[jaxls.Var[Array]],
        var_idx: jax.Array | int,
        kin: JaxKinTree,
        target_joint_indices: jax.Array,
        weights: float,
    ) -> list[jaxls.Factor]:
        """Manipulability cost."""

        def manipulability_cost(
            vals: jaxls.VarValues,
            kin: JaxKinTree,
            var: jaxls.Var[Array],
            target_joint_idx: int,
        ) -> Array:
            joint_cfg: jax.Array = vals[var]
            manipulability = RobotFactors.manip_yoshikawa(
                kin, joint_cfg, target_joint_idx
            )
            return ((1 / manipulability + 1e-6) * weights).flatten()

        return [
            jaxls.Factor(
                manipulability_cost,
                (kin, JointVarType(var_idx), target_joint_idx),
            )
            for target_joint_idx in target_joint_indices
        ]

    @staticmethod
    def manip_yoshikawa(
        kin: JaxKinTree,
        cfg: Array,
        target_joint_idx: int,
    ) -> Array:
        """Manipulability, as the determinant of the Jacobian."""
        jacobian = jax.jacfwd(
            lambda cfg: jaxlie.SE3(kin.forward_kinematics(cfg)).translation()
        )(cfg)[target_joint_idx]
        return jnp.sqrt(jnp.linalg.det(jacobian @ jacobian.T))
