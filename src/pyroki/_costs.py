import jax
from jax import Array
import jax.numpy as jnp
import jaxlie
import jaxls

from ._robot import Robot
from ._solver import CostFactor


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
        residual = (pose @ target_pose.inverse()).log()
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

        residual = (T_world_target_desired @ T_world_target.inverse()).log()
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
        return jnp.maximum(0.0, jnp.abs(joint_vel_eff) - robot.joint.velocity_limits_eff)


class RestCost(CostFactor[jaxls.Var[Array]]):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[Array],
    ) -> Array:
        """Bias towards joints at rest pose."""
        default = joint_var.default_factory()
        return vals[joint_var] - default


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
        """Manipulability cost."""
        manipulability = self.manip_yoshikawa(
            robot, vals[joint_var], target_joint_indices
        )
        return 1 / (manipulability + 1e-6)

    @staticmethod
    def manip_yoshikawa(
        robot: Robot,
        cfg: Array,
        target_joint_idx: jax.Array,
    ) -> Array:
        """Manipulability, as the determinant of the Jacobian."""
        jacobian = jax.jacfwd(
            lambda cfg: jaxlie.SE3(robot.forward_kinematics(cfg)).translation()
        )(cfg)
        jacobian = jacobian[target_joint_idx].squeeze()
        return jnp.sqrt(jnp.linalg.det(jnp.einsum("ij,ik->jk", jacobian, jacobian)))
