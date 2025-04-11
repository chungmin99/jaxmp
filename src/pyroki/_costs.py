import jax
from jax import Array
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jax import numpy as jnp
from jaxtyping import Float

from . import optim
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
class PoseCost(CostFactor[optim.Var[Array], jaxlie.SE3]):
    robot: Robot
    target_joint_indices: Array

    def cost_fn(
        self,
        vals: optim.VarValues,
        joint_var: optim.Var[Array],
        target_pose: jaxlie.SE3,
    ) -> Array:
        """Pose cost."""
        joint_cfg = vals[joint_var]
        Ts_joint_world = self.robot.forward_kinematics(joint_cfg)
        pose = jaxlie.SE3(Ts_joint_world[self.target_joint_indices])
        residual = (pose.inverse() @ target_pose).log()
        return residual


@jdc.pytree_dataclass
class PoseCostWithBase(CostFactor[optim.Var[Array], optim.Var[jaxlie.SE3], jaxlie.SE3]):
    robot: Robot
    target_joint_indices: Array

    def cost_fn(
        self,
        vals: optim.VarValues,
        joint_var: optim.Var[Array],
        T_world_base_var: optim.Var[jaxlie.SE3],
        T_world_target: jaxlie.SE3,
    ) -> Array:
        """Pose cost with base."""
        joint_cfg = vals[joint_var]
        T_world_base = vals[T_world_base_var]
        Ts_joint_world = self.robot.forward_kinematics(joint_cfg)
        T_base_target = jaxlie.SE3(Ts_joint_world[self.target_joint_indices])
        T_world_target_desired = T_world_base @ T_base_target

        residual = (T_world_target_desired.inverse() @ T_world_target).log()
        return residual


@jdc.pytree_dataclass
class LimitCost(CostFactor[optim.Var[Array]]):
    robot: Robot

    def cost_fn(
        self,
        vals: optim.VarValues,
        joint_var: optim.Var[Array],
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
class LimitVelCost(CostFactor[optim.Var[Array], optim.Var[Array]]):
    robot: Robot
    dt: float

    def cost_fn(
        self,
        vals: optim.VarValues,
        joint_var: optim.Var[Array],
        prev_joint_var: optim.Var[Array],
    ) -> Array:
        """Joint limit velocity cost."""
        joint_vel = (vals[joint_var] - vals[prev_joint_var]) / self.dt
        joint_vel_eff = self.robot.joint.get_full_derivative(joint_vel)
        return jnp.maximum(
            0.0, jnp.abs(joint_vel_eff) - self.robot.joint.velocity_limits_eff
        )


@jdc.pytree_dataclass
class RestCost(CostFactor[optim.Var[Array]]):
    rest_pose: Array

    def cost_fn(
        self,
        vals: optim.VarValues,
        joint_var: optim.Var[Array],
    ) -> Array:
        """Bias towards joints at the specified rest pose."""
        return vals[joint_var] - self.rest_pose


@jdc.pytree_dataclass
class RestCostWithBase(CostFactor[optim.Var[Array], optim.Var[jaxlie.SE3]]):
    rest_pose: Array

    def cost_fn(
        self,
        vals: optim.VarValues,
        joint_var: optim.Var[Array],
        T_world_base_var: optim.Var[jaxlie.SE3],
    ) -> Array:
        """Bias towards joints at the specified rest pose and identity base pose."""
        residual_joints = vals[joint_var] - self.rest_pose
        residual_base = vals[T_world_base_var].log()
        return jnp.concatenate([residual_joints, residual_base])


@jdc.pytree_dataclass
class SmoothnessCost(CostFactor[optim.Var[Array], optim.Var[Array]]):
    def cost_fn(
        self,
        vals: optim.VarValues,
        curr_joint_var: optim.Var[Array],
        past_joint_var: optim.Var[Array],
    ) -> Array:
        """Smoothness cost, for trajectories etc."""
        return vals[curr_joint_var] - vals[past_joint_var]


@jdc.pytree_dataclass
class ManipulabilityCost(CostFactor[optim.Var[Array]]):
    robot: Robot
    target_joint_indices: Array

    def cost_fn(
        self,
        vals: optim.VarValues,
        joint_var: optim.Var[Array],
    ) -> Array:
        """Manipulability cost (translation only).

        Sums the inverse manipulability across potentially multiple target indices.
        """
        # Vmap over the target_joint_indices
        vmapped_manip_yoshikawa = jax.vmap(
            self.manip_yoshikawa, in_axes=(None, None, 0)
        )
        manipulabilities = vmapped_manip_yoshikawa(vals[joint_var])
        return 1 / (manipulabilities + 1e-6)

    def manip_yoshikawa(
        self,
        cfg: Array,
    ) -> Array:
        """Manipulability, as the determinant of the Jacobian (translation only)."""
        jacobian = jax.jacfwd(
            lambda cfg: jaxlie.SE3(self.robot.forward_kinematics(cfg)).translation()
        )(cfg)
        jacobian = jacobian[self.target_joint_indices].squeeze()
        JJT = jacobian @ jacobian.T
        assert JJT.shape == (3, 3)
        return jnp.sqrt(jnp.linalg.det(JJT))


@jdc.pytree_dataclass
class SelfCollisionCost(CostFactor[optim.Var[Array]]):
    robot: Robot
    robot_coll: RobotCollision
    margin: float

    def cost_fn(
        self,
        vals: optim.VarValues,
        joint_var: optim.Var[Array],
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
class WorldCollisionCost(CostFactor[optim.Var[Array]]):
    robot: Robot
    robot_coll: RobotCollision
    margin: float
    world_geom: CollGeom

    def cost_fn(
        self,
        vals: optim.VarValues,
        joint_var: optim.Var[Array],
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
