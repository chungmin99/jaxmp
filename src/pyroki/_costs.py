import jax
from jax import Array
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie

import jaxls
from ._robot import Robot
from ._solver import CostFactor

from .coll._robot_collision import (
    RobotCollision,
    compute_self_collision_distance,
    compute_world_collision_distance,
)
from .coll._geometry import CollGeom
from .coll._collision import colldist_from_sdf


@jdc.pytree_dataclass
class PoseCost(CostFactor[jaxls.Var[Array], jaxlie.SE3]):
    robot: Robot
    target_link_indices: Array

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
        target_pose: jaxlie.SE3,
    ) -> Array:
        """Pose cost."""
        assert self.target_link_indices.dtype == jnp.int32
        joint_cfg = vals[joint_var]
        Ts_link_world = self.robot.forward_kinematics(joint_cfg)
        pose = jaxlie.SE3(Ts_link_world[self.target_link_indices])
        residual = (pose.inverse() @ target_pose).log()
        return residual


@jdc.pytree_dataclass
class PoseCostWithBase(CostFactor[jaxls.Var[Array], jaxls.Var[jaxlie.SE3], jaxlie.SE3]):
    robot: Robot
    target_link_indices: Array

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
        T_world_base_var: jaxls.Var[jaxlie.SE3],
        T_world_target: jaxlie.SE3,
    ) -> Array:
        """Pose cost with base."""
        assert self.target_link_indices.dtype == jnp.int32
        joint_cfg = vals[joint_var]
        T_world_base = vals[T_world_base_var]
        Ts_link_world = self.robot.forward_kinematics(joint_cfg)
        T_base_target_link = jaxlie.SE3(Ts_link_world[self.target_link_indices])
        T_world_target_link_actual = T_world_base @ T_base_target_link

        residual = (T_world_target_link_actual.inverse() @ T_world_target).log()
        return residual


@jdc.pytree_dataclass
class LimitCost(CostFactor[jaxls.Var[Array]]):
    robot: Robot

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
    ) -> Array:
        """Limit cost."""
        joint_cfg = vals[joint_var]
        joint_cfg_eff = self.robot.joint.get_full_config(joint_cfg)
        residual_upper = jnp.maximum(
            0.0, joint_cfg_eff - self.robot.joint.upper_limits_eff
        )
        residual_lower = jnp.maximum(
            0.0, self.robot.joint.lower_limits_eff - joint_cfg_eff
        )
        return residual_upper + residual_lower


@jdc.pytree_dataclass
class LimitVelCost(CostFactor[jaxls.Var[Array], jaxls.Var[Array]]):
    robot: Robot
    dt: float

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
        prev_joint_var: jaxls.Var[Array],
    ) -> Array:
        """Joint limit velocity cost."""
        joint_vel = (vals[joint_var] - vals[prev_joint_var]) / self.dt
        joint_vel_eff = self.robot.joint.get_full_derivative(joint_vel)
        return jnp.maximum(
            0.0, jnp.abs(joint_vel_eff) - self.robot.joint.velocity_limits_eff
        )


@jdc.pytree_dataclass
class RestCost(CostFactor[jaxls.Var[Array]]):
    rest_pose: Array

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
    ) -> Array:
        """Bias towards joints at the specified rest pose."""
        return vals[joint_var] - self.rest_pose


@jdc.pytree_dataclass
class RestCostWithBase(CostFactor[jaxls.Var[Array], jaxls.Var[jaxlie.SE3]]):
    rest_pose: Array

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
        T_world_base_var: jaxls.Var[jaxlie.SE3],
    ) -> Array:
        """Bias towards joints at the specified rest pose and identity base pose."""
        residual_joints = vals[joint_var] - self.rest_pose
        residual_base = vals[T_world_base_var].log()
        return jnp.concatenate([residual_joints, residual_base])


@jdc.pytree_dataclass
class SmoothnessCost(CostFactor[jaxls.Var[Array], jaxls.Var[Array]]):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        curr_joint_var: jaxls.Var[Array],
        past_joint_var: jaxls.Var[Array],
    ) -> Array:
        """Smoothness cost, for trajectories etc."""
        return vals[curr_joint_var] - vals[past_joint_var]


@jdc.pytree_dataclass
class ManipulabilityCost(CostFactor[jaxls.Var[Array]]):
    robot: Robot
    target_link_indices: Array

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
    ) -> Array:
        """Manipulability cost (translation only).

        Sums the inverse manipulability across potentially multiple target indices.
        """
        vmapped_manip_yoshikawa = jax.vmap(self.manip_yoshikawa, in_axes=(None, 0))
        manipulabilities = vmapped_manip_yoshikawa(
            vals[joint_var], self.target_link_indices
        )
        return 1 / (manipulabilities + 1e-6)

    def manip_yoshikawa(
        self,
        cfg: Array,
        target_link_index: jax.Array,
    ) -> Array:
        """Manipulability, as the determinant of the Jacobian (translation only)."""
        jacobian_all_links = jax.jacfwd(
            lambda q: jaxlie.SE3(self.robot.forward_kinematics(q)).translation()
        )(cfg)
        jacobian = jacobian_all_links[target_link_index]
        JJT = jacobian @ jacobian.T
        assert JJT.shape == (3, 3)
        return jnp.sqrt(jnp.linalg.det(JJT))


@jdc.pytree_dataclass
class SelfCollisionCost(CostFactor[jaxls.Var[Array]]):
    robot: Robot
    robot_coll: RobotCollision
    margin: float

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
    ) -> Array:
        """Cost penalizing self-collisions below a margin using smooth activation.

        Returns a cost vector, one entry for each active collision pair.
        Cost = colldist_from_sdf(distance, margin).
        Cost is >= 0.
        """
        cfg = vals[joint_var]
        active_distances = compute_self_collision_distance(
            self.robot_coll, self.robot, cfg
        )
        residual = colldist_from_sdf(active_distances, self.margin)
        return residual


@jdc.pytree_dataclass
class WorldCollisionCost(CostFactor[jaxls.Var[Array]]):
    robot: Robot
    robot_coll: RobotCollision
    margin: float
    world_geom: CollGeom

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
    ) -> Array:
        """Cost penalizing world collisions below a margin using smooth activation.

        Returns a cost matrix, shape (..., num_links, num_world_objects).
        Cost = colldist_from_sdf(distance, margin).
        """
        cfg = vals[joint_var]
        dist_matrix = compute_world_collision_distance(
            self.robot_coll, self.robot, cfg, self.world_geom
        )
        residual = colldist_from_sdf(dist_matrix, self.margin)
        return residual


@jdc.pytree_dataclass
class FivePointVelocityCost(
    CostFactor[
        jaxls.Var[Array],  # t+2
        jaxls.Var[Array],  # t+1
        jaxls.Var[Array],  # t-1
        jaxls.Var[Array],  # t-2
    ]
):
    """Cost penalizing deviation from joint velocity limits using a five-point stencil.

    Requires `robot.joint.velocity_limits_act` to be defined.
    Should be applied to variable indices `t = 2` to `N-3` for a trajectory of length `N`.
    """

    robot: Robot
    dt: float

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        var_t_plus_2: jaxls.Var[Array],
        var_t_plus_1: jaxls.Var[Array],
        var_t_minus_1: jaxls.Var[Array],
        var_t_minus_2: jaxls.Var[Array],
    ) -> Array:
        q_tm2, q_tm1, q_tp1, q_tp2 = (
            vals[var_t_minus_2],
            vals[var_t_minus_1],
            vals[var_t_plus_1],
            vals[var_t_plus_2],
        )
        velocity = (-q_tp2 + 8 * q_tp1 - 8 * q_tm1 + q_tm2) / (12 * self.dt)
        try:
            # Use actuated velocity limits
            vel_limits = self.robot.joint.velocity_limits_act
        except AttributeError:
            # Fallback if limits are not defined (cost becomes zero)
            vel_limits = jnp.inf
        limit_violation = jnp.maximum(0.0, jnp.abs(velocity) - vel_limits)
        return limit_violation


@jdc.pytree_dataclass
class FivePointAccelerationCost(
    CostFactor[
        jaxls.Var[Array],  # t
        jaxls.Var[Array],  # t+2
        jaxls.Var[Array],  # t+1
        jaxls.Var[Array],  # t-1
        jaxls.Var[Array],  # t-2
    ]
):
    """Cost minimizing joint acceleration using a five-point stencil.

    Should be applied to variable indices `t = 2` to `N-3` for a trajectory of length `N`.
    """

    dt: float

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        var_t: jaxls.Var[Array],
        var_t_plus_2: jaxls.Var[Array],
        var_t_plus_1: jaxls.Var[Array],
        var_t_minus_1: jaxls.Var[Array],
        var_t_minus_2: jaxls.Var[Array],
    ) -> Array:
        q_tm2, q_tm1, q_t, q_tp1, q_tp2 = (
            vals[var_t_minus_2],
            vals[var_t_minus_1],
            vals[var_t],
            vals[var_t_plus_1],
            vals[var_t_plus_2],
        )
        acceleration = (-q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2) / (
            12 * self.dt**2
        )
        return acceleration


@jdc.pytree_dataclass
class FivePointJerkCost(
    CostFactor[
        jaxls.Var[Array],  # t+3
        jaxls.Var[Array],  # t+2
        jaxls.Var[Array],  # t+1
        jaxls.Var[Array],  # t-1
        jaxls.Var[Array],  # t-2
        jaxls.Var[Array],  # t-3
    ]
):
    """Cost minimizing joint jerk using a seven-point stencil.

    Should be applied to variable indices `t = 3` to `N-4` for a trajectory of length `N`.
    """

    dt: float

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        var_t_plus_3: jaxls.Var[Array],
        var_t_plus_2: jaxls.Var[Array],
        var_t_plus_1: jaxls.Var[Array],
        var_t_minus_1: jaxls.Var[Array],
        var_t_minus_2: jaxls.Var[Array],
        var_t_minus_3: jaxls.Var[Array],
    ) -> Array:
        q_tm3, q_tm2, q_tm1, q_tp1, q_tp2, q_tp3 = (
            vals[var_t_minus_3],
            vals[var_t_minus_2],
            vals[var_t_minus_1],
            vals[var_t_plus_1],
            vals[var_t_plus_2],
            vals[var_t_plus_3],
        )
        jerk = (-q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3) / (
            8 * self.dt**3
        )
        return jerk
