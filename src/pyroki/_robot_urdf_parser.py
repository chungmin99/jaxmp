"""URDF parsing utilities for Robot class."""

import jax_dataclasses as jdc
import yourdfpy
from loguru import logger

from copy import deepcopy
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int
import jaxlie


@jdc.pytree_dataclass
class JointInfo:
    """Contains joint-related information for a robot."""

    count: jdc.Static[int]
    actuated_count: jdc.Static[int]
    twists: Float[Array, "actuated_count 6"]
    parent_transforms: Float[Array, "count 7"]
    parent_indices: Int[Array, " count"]
    actuated_indices: Int[Array, " count"]
    lower_limits: Float[Array, " actuated_count"]
    upper_limits: Float[Array, " actuated_count"]
    names: jdc.Static[tuple[str, ...]]
    velocity_limits: Float[Array, " actuated_count"]

    topo_sort_inv: Int[Array, " count"]
    """Inverse topological sort order, mapping sorted joint index to original joint index."""


@jdc.pytree_dataclass
class LinkInfo:
    """Contains link-related information for a robot."""

    count: jdc.Static[int]
    names: jdc.Static[tuple[str, ...]]
    parent_joint_indices: Int[Array, " count"]


class RobotURDFParser:
    """Parser for creating Robot instances from URDF files."""

    @staticmethod
    def _topologically_sort_joints(
        urdf: yourdfpy.URDF,
    ) -> Int[Array, " joints"]:
        """Calculates the topological processing order for joints and actuated joints.

        Ensures joints are processed parent-first for kinematic calculations, respecting
        mimic joint dependencies.

        Returns:
            - joint_order: Array of original joint indices sorted topologically.
        """
        original_joints = list(urdf.joint_map.values())
        original_act_joints = list(urdf.actuated_joints)
        num_joints = len(original_joints)
        num_act_joints = len(original_act_joints)

        original_name_to_idx = {j.name: i for i, j in enumerate(original_joints)}
        original_act_name_to_idx = {
            j.name: i for i, j in enumerate(original_act_joints)
        }

        # Perform topological sort based on parent-child and mimic relationships.
        joints_to_sort = deepcopy(original_joints)
        sorted_joint_objects = list[
            yourdfpy.Joint
        ]()  # Temporarily store sorted objects
        parent_link_of_joint = {j.child: j.parent for j in joints_to_sort}
        child_link_of_joint = {j.name: j.child for j in joints_to_sort}
        mimic_map = {
            j.name: j.mimic.joint for j in joints_to_sort if j.mimic is not None
        }

        processed_child_links = set()
        processed_joint_names = set()

        while len(sorted_joint_objects) < num_joints:
            found_next = False
            for i, j in enumerate(joints_to_sort):
                parent_link = parent_link_of_joint.get(j.child)

                # Check if parent link is ready
                parent_ok = (
                    parent_link == urdf.base_link
                    or parent_link in processed_child_links
                )
                # Check if mimic dependency is met
                mimic_ok = (j.name not in mimic_map) or (
                    mimic_map[j.name] in processed_joint_names
                )

                if parent_ok and mimic_ok:
                    sorted_joint_objects.append(j)
                    processed_child_links.add(child_link_of_joint[j.name])
                    processed_joint_names.add(j.name)
                    joints_to_sort.pop(i)
                    found_next = True
                    break
            if not found_next:
                # Simplified error handling for brevity during refactor
                remaining_names = [j.name for j in joints_to_sort]
                raise ValueError(
                    f"Topological sort failed. Remaining: {remaining_names}"
                )

        # Generate the topological order based on original indices
        joint_order = jnp.array(
            [original_name_to_idx[j.name] for j in sorted_joint_objects],
            dtype=jnp.int32,
        )

        # Generate topological order for actuated joints based on original indices
        act_joint_order_list = [
            original_act_name_to_idx[j.name]
            for j in sorted_joint_objects
            if j.name in original_act_name_to_idx
        ]

        # Ensure the count matches
        assert len(act_joint_order_list) == num_act_joints, (
            f"Mismatch in actuated joint count during topological sort. "
            f"Expected {num_act_joints}, found {len(act_joint_order_list)}"
        )

        return joint_order

    @staticmethod
    def parse(urdf: yourdfpy.URDF) -> tuple[JointInfo, LinkInfo]:
        """Build joint and link information from a URDF in the original order."""
        joint_twists_list = list[Array]()
        parent_transform_list = list[Array]()
        parent_idx_list = list[int]()
        actuated_idx_list = list[int]()
        lower_limit_list = list[float]()
        upper_limit_list = list[float]()
        joint_name_list = list[str]()
        velocity_limit_list = list[float]()

        # Link information.
        link_name_list = list[str]()
        parent_joint_idx_list = list[int]()

        # First pass: collect joint information.
        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            # Get joint names.
            joint_name_list.append(joint.name)

            # Get the actuated joint index.
            act_idx = RobotURDFParser._get_act_joint_idx(urdf, joint, joint_idx)
            actuated_idx_list.append(act_idx)

            # Get the twist parameters for all actuated joints.
            if joint in urdf.actuated_joints:
                twist = RobotURDFParser._get_act_joint_twist(joint)
                joint_twists_list.append(twist)

                # Get the joint limits.
                lower, upper = RobotURDFParser._get_joint_limits(joint)
                lower_limit_list.append(lower)
                upper_limit_list.append(upper)

                # Get the joint velocities.
                joint_vel_limit_val = RobotURDFParser._get_joint_limit_vel(joint)
                velocity_limit_list.append(joint_vel_limit_val)

            # Get the parent joint index and transform for each joint.
            parent_idx, T_parent_joint_val = RobotURDFParser._get_T_parent_joint(urdf, joint)
            parent_idx_list.append(parent_idx)
            parent_transform_list.append(T_parent_joint_val)

        # Second pass: collect link information.
        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            curr_link = joint.child
            # Ensure the link exists in the map before adding.
            if curr_link in urdf.link_map:
                link_name_list.append(curr_link)
                parent_joint_idx_list.append(joint_idx)

        # Calculate topological sort order
        topo_sort_inv_val = RobotURDFParser._topologically_sort_joints(urdf)

        # Create JointInfo and LinkInfo based on original order.
        joint_info = JointInfo(
            count=len(urdf.joint_map),
            actuated_count=len(urdf.actuated_joints),
            twists=jnp.array(joint_twists_list),
            parent_transforms=jnp.array(parent_transform_list),
            parent_indices=jnp.array(parent_idx_list, dtype=jnp.int32),
            actuated_indices=jnp.array(actuated_idx_list, dtype=jnp.int32),
            lower_limits=jnp.array(lower_limit_list),
            upper_limits=jnp.array(upper_limit_list),
            names=tuple(joint_name_list),
            velocity_limits=jnp.array(velocity_limit_list),
            topo_sort_inv=topo_sort_inv_val,
        )
        assert joint_info.twists.shape == (joint_info.actuated_count, 6)
        assert joint_info.parent_transforms.shape == (joint_info.count, 7)
        assert joint_info.parent_indices.shape == (joint_info.count,)
        assert joint_info.actuated_indices.shape == (joint_info.count,)
        assert joint_info.lower_limits.shape == (joint_info.actuated_count,)
        assert joint_info.upper_limits.shape == (joint_info.actuated_count,)
        assert joint_info.velocity_limits.shape == (joint_info.actuated_count,)
        assert joint_info.topo_sort_inv.shape == (joint_info.count,)

        link_info = LinkInfo(
            count=len(link_name_list),
            names=tuple(link_name_list),
            parent_joint_indices=jnp.array(parent_joint_idx_list, dtype=jnp.int32),
        )
        assert link_info.parent_joint_indices.shape == (link_info.count,)
        return joint_info, link_info

    @staticmethod
    def _get_act_joint_idx(
        urdf: yourdfpy.URDF, joint: yourdfpy.Joint, joint_idx: int
    ) -> int:
        """Get the original actuated joint index for a joint."""
        # Check if this joint is a mimic joint -- assume multiplier=1.0, offset=0.0.
        if joint.mimic is not None:
            mimicked_joint = urdf.joint_map[joint.mimic.joint]
            # Check if the mimicked joint itself is actuated.
            assert mimicked_joint in urdf.actuated_joints
            # Return the *original index* of the mimicked actuated joint
            mimicked_joint_idx = urdf.actuated_joints.index(mimicked_joint)
            act_joint_idx = mimicked_joint_idx

        # Track joint twists for actuated joints.
        elif joint in urdf.actuated_joints:
            assert joint.axis.shape == (3,)
            # Return the *original index* of this actuated joint
            act_joint_idx = urdf.actuated_joints.index(joint)

        # Not actuated.
        else:
            act_joint_idx = -1  # Represents non-actuated

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
    ) -> tuple[int, Array]:
        """Get the transform from the parent joint to the current joint,
        as well as the parent joint index."""
        assert joint.origin.shape == (4, 4)

        joint_from_child = {j.child: j for j in urdf.joint_map.values()}
        joint_name_to_idx = {j.name: i for i, j in enumerate(urdf.joint_map.values())}

        T_parent_joint = joint.origin
        if joint.parent not in joint_from_child:
            # Must be root node's joint (parent is base_link).
            parent_index = -1
        else:
            parent_joint = joint_from_child[joint.parent]
            parent_index = joint_name_to_idx[parent_joint.name]

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
