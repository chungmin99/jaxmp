import jax
from jax import Array
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
from typing import Tuple

from ._robot import Robot
from ._solver import CostFactor

from .coll._robot_collision import (
    RobotCollision,
    compute_self_collision_distance,
    compute_world_collision_distance,
)
from .coll._geometry import CollGeom
from .coll._collision import colldist_from_sdf


class PoseCost(CostFactor[Robot, jaxls.Var[Array], jaxlie.SE3, Array]):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        robot: Robot,
        joint_var: jaxls.Var[Array],
        target_pose: jaxlie.SE3,
        target_joint_indices: Array,
    ) -> Array:
        """Pose cost."""
        joint_cfg = vals[joint_var]
        Ts_joint_world = robot.forward_kinematics(joint_cfg)
        pose = jaxlie.SE3(Ts_joint_world[target_joint_indices])
        residual = (pose.inverse() @ target_pose).log()
        return residual


class PoseCostWithBase(
    CostFactor[Robot, jaxls.Var[Array], jaxls.Var[jaxlie.SE3], jaxlie.SE3, Array]
):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        robot: Robot,
        joint_var: jaxls.Var[Array],
        T_world_base_var: jaxls.Var[jaxlie.SE3],
        T_world_target: jaxlie.SE3,
        target_joint_indices: Array,
    ) -> Array:
        """Pose cost with base."""
        joint_cfg = vals[joint_var]
        T_world_base = vals[T_world_base_var]
        Ts_joint_world = robot.forward_kinematics(joint_cfg)
        T_base_target = jaxlie.SE3(Ts_joint_world[target_joint_indices])
        T_world_target_desired = T_world_base @ T_base_target

        residual = (T_world_target_desired.inverse() @ T_world_target).log()
        return residual


class LimitCost(CostFactor[Robot, jaxls.Var[Array]]):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        robot: Robot,
        joint_var: jaxls.Var[Array],
    ) -> Array:
        """Limit cost."""
        joint_cfg = vals[joint_var]
        joint_cfg_eff = robot.joint.get_full_config(joint_cfg)
        residual_upper = jnp.maximum(0.0, joint_cfg_eff - robot.joint.upper_limits_eff)
        residual_lower = jnp.maximum(0.0, robot.joint.lower_limits_eff - joint_cfg_eff)
        return residual_upper + residual_lower


class LimitVelCost(CostFactor[Robot, jaxls.Var[Array], jaxls.Var[Array], float]):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        robot: Robot,
        joint_var: jaxls.Var[Array],
        prev_joint_var: jaxls.Var[Array],
        dt: float,
    ) -> Array:
        """Joint limit velocity cost."""
        joint_vel = (vals[joint_var] - vals[prev_joint_var]) / dt
        joint_vel_eff = robot.joint.get_full_derivative(joint_vel)
        return jnp.maximum(
            0.0, jnp.abs(joint_vel_eff) - robot.joint.velocity_limits_eff
        )


class RestCost(CostFactor[jaxls.Var[Array], Array]):
    """Cost factor that penalizes deviation from a specified rest pose."""

    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
        rest_pose: Array,
    ) -> Array:
        """Bias towards joints at the specified rest pose."""
        return vals[joint_var] - rest_pose


class RestCostWithBase(CostFactor[jaxls.Var[Array], jaxls.Var[jaxlie.SE3], Array]):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
        T_world_base_var: jaxls.Var[jaxlie.SE3],
        rest_pose: Array,
    ) -> Array:
        """Bias towards joints at the specified rest pose and identity base pose."""
        residual_joints = vals[joint_var] - rest_pose
        residual_base = vals[T_world_base_var].log()
        return jnp.concatenate([residual_joints, residual_base])


class SmoothnessCost(CostFactor[jaxls.Var[Array], jaxls.Var[Array]]):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        curr_joint_var: jaxls.Var[Array],
        past_joint_var: jaxls.Var[Array],
    ) -> Array:
        """Smoothness cost, for trajectories etc."""
        return vals[curr_joint_var] - vals[past_joint_var]


class ManipulabilityCost(CostFactor[Robot, jaxls.Var[Array], Array]):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        robot: Robot,
        joint_var: jaxls.Var[Array],
        target_joint_indices: Array,
    ) -> Array:
        """Manipulability cost (translation only).

        Sums the inverse manipulability across potentially multiple target indices.
        """
        # Vmap over the target_joint_indices
        assert len(target_joint_indices.shape) == 1
        vmapped_manip_yoshikawa = jax.vmap(
            self.manip_yoshikawa, in_axes=(None, None, 0)
        )
        manipulabilities = vmapped_manip_yoshikawa(
            robot, vals[joint_var], target_joint_indices
        )
        return 1 / (manipulabilities + 1e-6)

    @staticmethod
    def manip_yoshikawa(
        robot: Robot,
        cfg: Array,
        target_joint_idx: jax.Array,
    ) -> Array:
        """Manipulability, as the determinant of the Jacobian (translation only)."""
        jacobian = jax.jacfwd(
            lambda cfg: jaxlie.SE3(robot.forward_kinematics(cfg)).translation()
        )(cfg)
        jacobian = jacobian[target_joint_idx].squeeze()
        JJT = jacobian @ jacobian.T
        assert JJT.shape == (3, 3)
        return jnp.sqrt(jnp.linalg.det(JJT))


class SelfCollisionCost(CostFactor[Robot, RobotCollision, jaxls.Var[Array], float]):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        robot: Robot,
        robot_coll: RobotCollision,
        joint_var: jaxls.Var[Array],
        margin: float,
    ) -> Array:
        """Cost penalizing self-collisions below a margin using smooth activation.

        Returns a cost vector, one entry for each active collision pair.
        Cost = colldist_from_sdf(distance, margin).
        Cost is >= 0.
        """
        cfg = vals[joint_var]
        active_distances = compute_self_collision_distance(robot_coll, robot, cfg)
        residual = colldist_from_sdf(active_distances, margin)
        return residual


class WorldCollisionCost(
    CostFactor[Robot, RobotCollision, jaxls.Var[Array], CollGeom, float]
):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        robot: Robot,
        robot_coll: RobotCollision,
        joint_var: jaxls.Var[Array],
        world_geom: CollGeom,
        margin: float,
    ) -> Array:
        """Cost penalizing world collisions below a margin using smooth activation.

        Returns a cost matrix, shape (..., num_links, num_world_objects).
        Cost = colldist_from_sdf(distance, margin).
        """
        cfg = vals[joint_var]
        dist_matrix = compute_world_collision_distance(
            robot_coll, robot, cfg, world_geom
        )
        residual = colldist_from_sdf(dist_matrix, margin)
        return residual
