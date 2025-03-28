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


class LimitCost(CostFactor[Robot, jaxls.Var[Array]]):
    def cost_fn(
        self,
        vals: jaxls.VarValues,
        robot: Robot,
        joint_var: jaxls.Var[Array],
    ) -> Array:
        """Limit cost."""
        joint_cfg = vals[joint_var]
        residual_upper = jnp.maximum(0.0, joint_cfg - robot.limits_upper)
        residual_lower = jnp.maximum(0.0, robot.limits_lower - joint_cfg)
        return residual_upper + residual_lower
