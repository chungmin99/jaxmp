# pylint: disable=invalid-name

from __future__ import annotations

from typing import Optional

import jax
import jax_dataclasses as jdc
import jaxlie
import yourdfpy
from loguru import logger

from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

import jaxls

from ._robot_urdf_parser import JointInfo, LinkInfo, RobotURDFParser


@jdc.pytree_dataclass
class Robot:
    """A differentiable robot kinematics tree."""

    joint_info: JointInfo
    """Joint information for the robot."""

    link_info: LinkInfo
    """Link information for the robot."""

    unroll_fk: jdc.Static[bool]
    """Whether to unroll the forward kinematics `fori_loop`."""

    JointVar: jdc.Static[type[jaxls.Var[Array]]]
    """Variable class for the robot configuration."""

    _act_joint_sort_order: Float[Array, " act_joints"]
    """Sort order for the actuated joints, used for undoing the topological sort."""

    _joint_sort_order: Float[Array, " joints"]
    """Sort order for the joints, used for undoing the topological sort."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        unroll_fk: bool = False,
        *,
        act_joint_sort_order: Optional[Float[Array, " act_joints"]] = None,
        joint_sort_order: Optional[Float[Array, " joints"]] = None,
    ) -> Robot:
        joint_info, link_info = RobotURDFParser.parse(urdf)

        if act_joint_sort_order is None:
            act_joint_sort_order = jnp.arange(joint_info.num_actuated_joints)
        if joint_sort_order is None:
            joint_sort_order = jnp.arange(joint_info.num_joints)

        JointVar = Robot.get_joint_var_class(
            default_val=(joint_info.limits_lower + joint_info.limits_upper) / 2,
            num_actuated_joints=joint_info.num_actuated_joints,
            joint_vel_limit=joint_info.joint_vel_limit,
        )

        return Robot(
            joint_info=joint_info,
            link_info=link_info,
            unroll_fk=unroll_fk,
            JointVar=JointVar,
            _act_joint_sort_order=act_joint_sort_order,
            _joint_sort_order=joint_sort_order,
        )

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch num_act_joints"],
    ) -> Float[Array, "*batch num_joints 7"]:
        """
        Run forward kinematics on the robot, in the provided configuration.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch num_act_joints)`.

        Returns:
            The SE(3) transforms of the joints, in the format `(*batch num_joints wxyz_xyz)`.
        """
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.joint_info.num_actuated_joints)

        Ts_joint_child = jaxlie.SE3.exp(
            self.joint_info.joint_twists * cfg[..., None]
        ).wxyz_xyz
        assert Ts_joint_child.shape == (
            *batch_axes,
            self.joint_info.num_actuated_joints,
            7,
        )

        Ts_joint_child = jnp.where(
            (self.joint_info.idx_actuated_joint == -1)[..., None],
            jaxlie.SE3.identity().wxyz_xyz,
            Ts_joint_child[..., self.joint_info.idx_actuated_joint, :],
        )
        Ts_parent_child = (
            jaxlie.SE3(self.joint_info.Ts_parent_joint) @ jaxlie.SE3(Ts_joint_child)
        ).wxyz_xyz

        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                self.joint_info.idx_parent_joint[i] == -1,
                jaxlie.SE3.identity().wxyz_xyz,
                Ts_world_joint[..., self.joint_info.idx_parent_joint[i], :],
            )

            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent) @ jaxlie.SE3(Ts_parent_child[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_parent = jnp.zeros((*batch_axes, self.joint_info.num_joints, 7))
        Ts_world_joint = jax.lax.fori_loop(
            lower=0,
            upper=self.joint_info.num_joints,
            body_fun=compute_joint,
            init_val=Ts_world_parent,
            unroll=self.unroll_fk,
        )

        assert Ts_world_joint.shape == (*batch_axes, self.joint_info.num_joints, 7)
        return Ts_world_joint

    @jdc.jit
    def forward_kinematics_links(
        self,
        cfg: Float[Array, "*batch num_act_joints"],
    ) -> Float[Array, "*batch num_links 7"]:
        """Run forward kinematics on the robot's links, in the provided configuration."""
        Ts_world_joint = self.forward_kinematics(cfg)
        return Ts_world_joint[..., self.link_info.idx_link_parent, :]

    @staticmethod
    def get_joint_var_class(
        default_val: Float[Array, " num_act_joints"],
        num_actuated_joints: int,
        joint_vel_limit: Float[Array, " num_act_joints"],
    ) -> type[jaxls.Var[Array]]:
        """Return a variable class for the robot configuration,
        considering different joint units for revolute/prismatic joints."""

        @jdc.jit
        def retract_fn(
            cfg: Float[Array, "*batch num_act_joints"],
            delta: Float[Array, "*batch num_act_joints"],
        ) -> Float[Array, "*batch num_act_joints"]:
            """Retract function for the robot."""
            assert cfg.shape == delta.shape
            assert cfg.shape[-1] == num_actuated_joints

            # Apply units to delta, by normalizing w/ the joint velocity.
            # Important for robots with both revolute + prismatic joints
            # (e.g., fetch, robot grippers).
            _delta = delta * joint_vel_limit * 0.01

            return cfg + _delta

        class JointVar(  # pylint: disable=missing-class-docstring
            jaxls.Var[Array],
            default_factory=lambda: default_val.copy(),
            tangent_dim=num_actuated_joints,
            retract_fn=retract_fn,
        ): ...

        return JointVar
