"""URDF parsing utilities for Robot class."""

import jax_dataclasses as jdc
import yourdfpy
from loguru import logger

from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int
import jaxlie


@jdc.pytree_dataclass
class JointInfo:
    """Contains joint-related information for a robot."""

    num_joints: jdc.Static[int]
    num_actuated_joints: jdc.Static[int]
    joint_twists: Float[Array, "act_joints 6"]
    Ts_parent_joint: Float[Array, "joints 7"]
    idx_parent_joint: Int[Array, " joints"]
    idx_actuated_joint: Int[Array, " joints"]
    limits_lower: Float[Array, " act_joints"]
    limits_upper: Float[Array, " act_joints"]
    joint_names: jdc.Static[tuple[str, ...]]
    joint_vel_limit: Float[Array, " act_joints"]


@jdc.pytree_dataclass
class LinkInfo:
    """Contains link-related information for a robot."""

    num_links: jdc.Static[int]
    link_names: jdc.Static[tuple[str, ...]]
    idx_link_parent: Int[Array, " links"]


class RobotURDFParser:
    """Parser for creating Robot instances from URDF files."""

    @staticmethod
    def parse(urdf: yourdfpy.URDF) -> tuple[JointInfo, LinkInfo]:
        """Build joint and link information from a URDF."""
        # Get the parent indices + joint twist parameters.
        joint_twists = list[Array]()
        Ts_parent_joint = list[Array]()
        idx_parent_joint = list[int]()
        idx_actuated_joint = list[int]()
        limits_lower = list[float]()
        limits_upper = list[float]()
        joint_names = list[str]()
        joint_vel_limits = list[float]()

        # Link information
        link_names = list[str]()
        idx_link_parent = list[int]()

        # First pass: collect joint information
        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            # Get joint names.
            joint_names.append(joint.name)

            # Get the actuated joint index.
            act_idx = RobotURDFParser._get_act_joint_idx(urdf, joint, joint_idx)
            idx_actuated_joint.append(act_idx)

            # Get the twist parameters for all actuated joints.
            if joint in urdf.actuated_joints:
                twist = RobotURDFParser._get_act_joint_twist(joint)
                joint_twists.append(twist)

                # Get the joint limits.
                lower, upper = RobotURDFParser._get_joint_limits(joint)
                limits_lower.append(lower)
                limits_upper.append(upper)

                # Get the joint velocities.
                joint_vel_limit = RobotURDFParser._get_joint_limit_vel(joint)
                joint_vel_limits.append(joint_vel_limit)

            # Get the parent joint index and transform for each joint.
            parent_idx, T_parent_joint = RobotURDFParser._get_T_parent_joint(
                urdf, joint, joint_idx
            )
            idx_parent_joint.append(parent_idx)
            Ts_parent_joint.append(T_parent_joint)

        # Second pass: collect link information
        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            curr_link = joint.child
            assert curr_link in urdf.link_map
            
            link_names.append(curr_link)
            idx_link_parent.append(joint_idx)

        return (
            JointInfo(
                num_joints=len(urdf.joint_map),
                num_actuated_joints=len(urdf.actuated_joints),
                joint_twists=jnp.array(joint_twists),
                Ts_parent_joint=jnp.array(Ts_parent_joint),
                idx_parent_joint=jnp.array(idx_parent_joint),
                idx_actuated_joint=jnp.array(idx_actuated_joint),
                limits_lower=jnp.array(limits_lower),
                limits_upper=jnp.array(limits_upper),
                joint_names=tuple(joint_names),
                joint_vel_limit=jnp.array(joint_vel_limits),
            ),
            LinkInfo(
                num_links=len(link_names),
                link_names=tuple(link_names),
                idx_link_parent=jnp.array(idx_link_parent),
            ),
        )

    @staticmethod
    def _get_act_joint_idx(
        urdf: yourdfpy.URDF, joint: yourdfpy.Joint, joint_idx: int
    ) -> int:
        """Get the actuated joint index for a joint, checking for mimic joints."""
        # Check if this joint is a mimic joint -- assume multiplier=1.0, offset=0.0.
        if joint.mimic is not None:
            mimicked_joint = urdf.joint_map[joint.mimic.joint]
            mimicked_joint_idx = urdf.actuated_joints.index(mimicked_joint)
            assert mimicked_joint_idx < joint_idx, "Code + fk `fori_loop` assumes this!"
            logger.warning("Mimic joint detected.")
            act_joint_idx = urdf.actuated_joints.index(mimicked_joint)

        # Track joint twists for actuated joints.
        elif joint in urdf.actuated_joints:
            assert joint.axis.shape == (3,)
            act_joint_idx = urdf.actuated_joints.index(joint)

        # Not actuated.
        else:
            act_joint_idx = -1

        return act_joint_idx

    @staticmethod
    def _get_act_joint_twist(joint: yourdfpy.Joint) -> Array:
        """Get the twist parameters for an actuated joint."""
        if joint.type in ("revolute", "continuous"):
            twist = jnp.concatenate([jnp.zeros(3), joint.axis])
        elif joint.type == "prismatic":
            twist = jnp.concatenate([joint.axis, jnp.zeros(3)])
        else:
            raise ValueError(f"Unsupported joint type {joint.type}!")
        return twist

    @staticmethod
    def _get_T_parent_joint(
        urdf: yourdfpy.URDF,
        joint: yourdfpy.Joint,
        joint_idx: int,
    ) -> tuple[int, Array]:
        """Get the transform from the parent joint to the current joint,
        as well as the parent joint index."""
        assert joint.origin.shape == (4, 4)

        joint_from_child = {joint.child: joint for joint in urdf.joint_map.values()}

        T_parent_joint = joint.origin
        if joint.parent not in joint_from_child:
            # Must be root node.
            parent_index = -1
        else:
            parent_joint = joint_from_child[joint.parent]
            parent_index = urdf.joint_names.index(parent_joint.name)
            if parent_index >= joint_idx:
                logger.warning(
                    f"Parent index {parent_index} >= joint index {joint_idx}! "
                    + "Assuming that parent is root."
                )
                if parent_joint.parent != urdf.scene.graph.base_frame:
                    raise ValueError(
                        "Parent index >= joint_index, but parent is not root!"
                    )
                T_parent_joint = parent_joint.origin @ T_parent_joint  # T_root_joint.
                parent_index = -1

        return (parent_index, jaxlie.SE3.from_matrix(T_parent_joint).wxyz_xyz)

    @staticmethod
    def _get_joint_limits(joint: yourdfpy.Joint) -> tuple[float, float]:
        """Get the joint limits for an actuated joint, returns (lower, upper)."""
        assert joint.limit is not None
        if joint.limit.lower is not None and joint.limit.upper is not None:
            lower = joint.limit.lower
            upper = joint.limit.upper
        elif joint.type == "continuous":
            logger.warning("Continuous joint detected, cap to [-pi, pi] limits.")
            lower = -jnp.pi
            upper = jnp.pi
        else:
            raise ValueError("We currently assume there are joint limits!")
        return lower, upper

    @staticmethod
    def _get_joint_limit_vel(joint: yourdfpy.Joint) -> float:
        """Get the joint velocity for an actuated joint."""
        if joint.limit is not None and joint.limit.velocity is not None:
            return joint.limit.velocity
        logger.warning("Joint velocity not specified, defaulting to 1.0.")
        return 1.0
