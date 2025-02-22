"""
Differentiable robot kinematics model, implemented in JAX.
Includes:
 - URDF parsing.
 - Forward kinematics.
 - Support for mimic joints.
"""

# pylint: disable=invalid-name

from __future__ import annotations

from typing import Callable

import jax
import jax_dataclasses as jdc
import jaxlie
import yourdfpy
from loguru import logger

from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int


@jdc.pytree_dataclass
class JaxKinTree:
    """A differentiable robot kinematics tree."""

    # Core joint parameters.
    num_joints: jdc.Static[int]
    """Number of joints in the robot."""
    num_actuated_joints: jdc.Static[int]
    """Number of actuated joints in the robot."""
    joint_names: jdc.Static[tuple[str]]
    """Names of the joints, in shape `joints`."""

    # Joint relationships and transforms.
    joint_twists: Float[Array, "joints 6"]
    """Twist parameters for each joint. Zero for fixed joints."""
    Ts_parent_joint: Float[Array, "joints 7"]
    """Transform from parent joint to current joint, in the format `joints 7`."""
    idx_parent_joint: Int[Array, " joints"]
    """Parent joint index for each joint. -1 for root."""
    idx_actuated_joint: Int[Array, " joints"]
    """Index of actuated joint for each joint, for handling mimic joints. -1 otherwise."""

    # Joint limits.
    limits_lower: Float[Array, " act_joints"]
    """Lower joint limits for each actuated joint."""
    limits_upper: Float[Array, " act_joints"]
    """Upper joint limits for each actuated joint."""
    _limits_lower_all: Float[Array, " joints"]
    """Lower joint limits for each joint; includes mimic and fixed joints."""
    _limits_upper_all: Float[Array, " joints"]
    """Upper joint limits for each joint; includes mimic and fixed joints."""

    # Velocity limits.
    joint_vel_limit: Float[Array, " act_joints"]
    """Joint limit velocities for each actuated joint."""
    _joint_vel_limit_all: Float[Array, " joints"]
    """Joint limit velocities for each joint; includes mimic and fixed joints."""

    # Mimic joint parameters.
    mimic_multiplier: Float[Array, " joints"]
    """Multiplier for mimic joints. 1.0 for non-mimic joints."""
    mimic_offset: Float[Array, " joints"]
    """Offset for mimic joints. 0.0 for non-mimic joints."""

    # Configuration.
    unroll_fk: jdc.Static[bool]
    """Whether to unroll the forward kinematics `fori_loop`."""

    @staticmethod
    def from_urdf(urdf: yourdfpy.URDF, unroll_fk: bool = False) -> JaxKinTree:
        """Build a differentiable robot model from a URDF."""
        # Initialize lists to store joint parameters.
        joint_twists = []
        Ts_parent_joint = []
        idx_parent_joint = []
        idx_actuated_joint = []
        limits_lower = []
        limits_upper = []
        limits_lower_all = []
        limits_upper_all = []
        joint_names = []
        joint_vel_limits_all = []
        joint_vel_limits = []
        mimic_multiplier = []
        mimic_offset = []

        # Process each joint in the URDF.
        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            # Get joint names.
            joint_names.append(joint.name)

            # Get the actuated joint index.
            act_idx = JaxKinTree._get_act_joint_idx(urdf, joint, joint_idx)
            idx_actuated_joint.append(act_idx)

            # Process movable joints (actuated or mimic).
            if joint in urdf.actuated_joints or joint.mimic is not None:
                twist = JaxKinTree._get_act_joint_twist(joint)
                joint_twists.append(twist)

                # Get joint limits.
                lower, upper = JaxKinTree._get_joint_limits(joint)
                if joint.mimic is None:
                    limits_lower.append(lower)
                    limits_upper.append(upper)
                limits_lower_all.append(lower)
                limits_upper_all.append(upper)

                # Get joint velocities.
                joint_vel_limit = JaxKinTree._get_joint_limit_vel(joint)
                joint_vel_limits_all.append(joint_vel_limit)
                if joint.mimic is None:
                    joint_vel_limits.append(joint_vel_limit)

                # Handle mimic joint parameters.
                if joint.mimic is not None:
                    multiplier = (
                        1.0
                        if joint.mimic.multiplier is None
                        else joint.mimic.multiplier
                    )
                    offset = 0.0 if joint.mimic.offset is None else joint.mimic.offset
                else:
                    multiplier = 1.0
                    offset = 0.0
                mimic_multiplier.append(multiplier)
                mimic_offset.append(offset)
            else:
                # Fixed joint handling.
                joint_twists.append(jnp.zeros(6))
                limits_lower_all.append(0)
                limits_upper_all.append(0)
                joint_vel_limits_all.append(0.0)
                mimic_multiplier.append(1.0)
                mimic_offset.append(0.0)

            # Get parent joint information.
            if joint.origin is None and joint.type == "fixed":
                # Fixed joint, no transform.
                logger.info("Found fixed joint with no origin, placing at origin.")
                parent_idx = -1
                T_parent_joint = jaxlie.SE3.identity().wxyz_xyz
            else:
                parent_idx, T_parent_joint = JaxKinTree._get_T_parent_joint(
                    urdf, joint, joint_idx
                )
            idx_parent_joint.append(parent_idx)
            Ts_parent_joint.append(T_parent_joint)

        # Convert lists to arrays.
        kin = JaxKinTree(
            num_joints=len(urdf.joint_map),
            num_actuated_joints=len(urdf.actuated_joints),
            idx_actuated_joint=jnp.array(idx_actuated_joint),
            joint_twists=jnp.array(joint_twists),
            Ts_parent_joint=jnp.array(Ts_parent_joint),
            idx_parent_joint=jnp.array(idx_parent_joint),
            limits_lower=jnp.array(limits_lower),
            limits_upper=jnp.array(limits_upper),
            _limits_lower_all=jnp.array(limits_lower_all),
            _limits_upper_all=jnp.array(limits_upper_all),
            joint_names=tuple(joint_names),
            joint_vel_limit=jnp.array(joint_vel_limits),
            _joint_vel_limit_all=jnp.array(joint_vel_limits_all),
            unroll_fk=unroll_fk,
            mimic_multiplier=jnp.array(mimic_multiplier),
            mimic_offset=jnp.array(mimic_offset),
        )

        # Shape assertions.
        assert kin.joint_twists.shape == (kin.num_joints, 6)
        assert kin.Ts_parent_joint.shape == (kin.num_joints, 7)
        assert kin.idx_parent_joint.shape == (kin.num_joints,)
        assert kin.idx_actuated_joint.shape == (kin.num_joints,)
        assert kin.limits_lower.shape == (kin.num_actuated_joints,)
        assert kin.limits_upper.shape == (kin.num_actuated_joints,)
        assert kin._limits_lower_all.shape == (kin.num_joints,)
        assert kin._limits_upper_all.shape == (kin.num_joints,)
        assert kin.joint_vel_limit.shape == (kin.num_actuated_joints,)
        assert kin._joint_vel_limit_all.shape == (kin.num_joints,)
        assert kin.mimic_multiplier.shape == (kin.num_joints,)
        assert kin.mimic_offset.shape == (kin.num_joints,)

        return kin

    @staticmethod
    def _get_act_joint_idx(
        urdf: yourdfpy.URDF, joint: yourdfpy.Joint, joint_idx: int
    ) -> int:
        """Get the actuated joint index for a joint, checking for mimic joints."""
        if joint.mimic is not None:
            mimicked_joint = urdf.joint_map[joint.mimic.joint]
            mimicked_joint_idx = urdf.actuated_joints.index(mimicked_joint)
            assert mimicked_joint_idx < joint_idx, "Code + fk `fori_loop` assumes this!"
            logger.warning("Mimic joint detected.")
            return urdf.actuated_joints.index(mimicked_joint)
        elif joint in urdf.actuated_joints:
            assert joint.axis.shape == (3,)
            return urdf.actuated_joints.index(joint)
        return -1

    @staticmethod
    def _get_act_joint_twist(joint: yourdfpy.Joint) -> Array:
        """Get the twist parameters for an actuated joint."""
        if joint.type in ("revolute", "continuous"):
            return jnp.concatenate([jnp.zeros(3), joint.axis])
        elif joint.type == "prismatic":
            return jnp.concatenate([joint.axis, jnp.zeros(3)])
        raise ValueError(f"Unsupported joint type {joint.type}!")

    @staticmethod
    def _get_T_parent_joint(
        urdf: yourdfpy.URDF,
        joint: yourdfpy.Joint,
        joint_idx: int,
    ) -> tuple[int, Array]:
        """Get the transform from the parent joint to the current joint and parent joint index."""
        assert joint.origin.shape == (4, 4)
        joint_from_child = {joint.child: joint for joint in urdf.joint_map.values()}

        T_parent_joint = joint.origin
        if joint.parent not in joint_from_child:
            return -1, jaxlie.SE3.from_matrix(T_parent_joint).wxyz_xyz

        parent_joint = joint_from_child[joint.parent]
        parent_index = urdf.joint_names.index(parent_joint.name)

        if parent_index >= joint_idx:
            logger.warning(
                f"Parent index {parent_index} >= joint index {joint_idx}! "
                "Assuming that parent is root."
            )
            if parent_joint.parent != urdf.scene.graph.base_frame:
                raise ValueError("Parent index >= joint_index, but parent is not root!")
            T_parent_joint = parent_joint.origin @ T_parent_joint  # T_root_joint
            parent_index = -1

        return parent_index, jaxlie.SE3.from_matrix(T_parent_joint).wxyz_xyz

    @staticmethod
    def _get_joint_limits(joint: yourdfpy.Joint) -> tuple[float, float]:
        """Get the joint limits for an actuated joint, returns (lower, upper)."""
        assert joint.limit is not None
        if joint.limit.lower is not None and joint.limit.upper is not None:
            return joint.limit.lower, joint.limit.upper
        elif joint.type == "continuous":
            logger.warning("Continuous joint detected, cap to [-pi, pi] limits.")
            return -jnp.pi, jnp.pi
        raise ValueError("We currently assume there are joint limits!")

    @staticmethod
    def _get_joint_limit_vel(joint: yourdfpy.Joint) -> float:
        """Get the joint velocity for an actuated joint."""
        if joint.limit is not None and joint.limit.velocity is not None:
            return joint.limit.velocity
        logger.warning("Joint velocity not specified, defaulting to 1.0.")
        return 1.0

    @jdc.jit
    def map_actuated_to_all_joints(
        self, cfg: Float[Array, "*batch num_act_joints"], apply_mimic_scale: jdc.Static[bool] = False
    ) -> Float[Array, "*batch num_joints"]:
        """Expand the actuated joint configuration to the full joint configuration.
        If `apply_mimic_scale` is True, the mimic multiplier/offset settings are applied."""
        batch_axes = cfg.shape[:-1]
        cfg = cfg[..., self.idx_actuated_joint]
        cfg = jnp.where(self.idx_actuated_joint == -1, 0.0, cfg)
        assert cfg.shape == (*batch_axes, self.num_joints)
        if apply_mimic_scale:
            cfg = cfg * self.mimic_multiplier + self.mimic_offset
        assert cfg.shape == (*batch_axes, self.num_joints)
        return cfg

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch num_act_joints"],
    ) -> Float[Array, "*batch num_joints 7"]:
        """Run forward kinematics on the robot, in the provided configuration.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch num_act_joints)`.

        Returns:
            The SE(3) transforms of the joints, in the format `(*batch num_joints wxyz_xyz)`.
        """
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.num_actuated_joints)

        # Map to full joint space and apply mimic scaling.
        cfg = self.map_actuated_to_all_joints(cfg, apply_mimic_scale=True)
        assert cfg.shape == (*batch_axes, self.num_joints)

        # Compute transforms for all joints.
        Ts_joint_child = jaxlie.SE3.exp(self.joint_twists * cfg[..., None]).wxyz_xyz
        assert Ts_joint_child.shape == (*batch_axes, self.num_joints, 7)

        Ts_parent_child = (
            jaxlie.SE3(self.Ts_parent_joint) @ jaxlie.SE3(Ts_joint_child)
        ).wxyz_xyz

        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                self.idx_parent_joint[i] == -1,
                jaxlie.SE3.identity().wxyz_xyz,
                Ts_world_joint[..., self.idx_parent_joint[i], :],
            )

            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent) @ jaxlie.SE3(Ts_parent_child[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_parent = jnp.zeros((*batch_axes, self.num_joints, 7))
        Ts_world_joint = jax.lax.fori_loop(
            lower=0,
            upper=self.num_joints,
            body_fun=compute_joint,
            init_val=Ts_world_parent,
            unroll=self.unroll_fk,
        )

        assert Ts_world_joint.shape == (*batch_axes, self.num_joints, 7)
        return Ts_world_joint

    def get_retract_fn(
        self,
    ) -> Callable[
        [Float[Array, "*batch num_act_joints"], Float[Array, "*batch num_act_joints"]],
        Float[Array, "*batch num_act_joints"],
    ]:
        """Return a retract function for the robot configuration,
        considering different joint units for revolute/prismatic joints."""

        @jdc.jit
        def retract_fn(
            cfg: Float[Array, "*batch num_act_joints"],
            delta: Float[Array, "*batch num_act_joints"],
        ) -> Float[Array, "*batch num_act_joints"]:
            """Retract function for the robot."""
            assert cfg.shape == delta.shape
            assert cfg.shape[-1] == self.num_actuated_joints

            # Apply units to delta, by normalizing w/ the joint velocity.
            # Important for robots with both revolute + prismatic joints.
            _delta = delta * self.joint_vel_limit * 0.01

            return cfg + _delta

        return retract_fn
