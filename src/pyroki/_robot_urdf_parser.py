"""URDF parsing utilities for Robot class."""

import jax_dataclasses as jdc
import yourdfpy
from loguru import logger

from copy import deepcopy
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int, Bool
import jaxlie


@jdc.pytree_dataclass
class JointInfo:
    """Contains joint-related information for a robot."""

    count: jdc.Static[int]
    actuated_count: jdc.Static[int]
    names: jdc.Static[tuple[str, ...]]

    # Joint parameters.
    twists: Float[Array, "n_joints 6"]
    """Twist parameters for each joint. Shape: (n_joints, 6)."""
    parent_transforms: Float[Array, "n_joints 7"]
    """Transform from parent joint to current joint. Shape: (n_joints, 7)."""
    parent_indices: Int[Array, " n_joints"]
    """Index of the parent joint for each joint. Shape: (n_joints,)."""
    actuated_indices: Int[Array, " n_joints"]

    # Limits for directly actuated joints.
    lower_limits_act: Float[Array, " n_act_joints"]
    """Lower limits for directly actuated joints. Shape: (n_act_joints,)."""
    upper_limits_act: Float[Array, " n_act_joints"]
    """Upper limits for directly actuated joints. Shape: (n_act_joints,)."""
    velocity_limits_act: Float[Array, " n_act_joints"]
    """Velocity limits for directly actuated joints. Shape: (n_act_joints,)."""

    # Effective limits for ALL joints (including mimics and fixed).
    lower_limits_eff: Float[Array, " n_joints"]
    """Lower limits for all joints. Shape: (n_joints,)."""
    upper_limits_eff: Float[Array, " n_joints"]
    """Upper limits for all joints. Shape: (n_joints,)."""
    velocity_limits_eff: Float[Array, " n_joints"]
    """Velocity limits for all joints. Shape: (n_joints,)."""

    # Mimic joint parameters.
    mimic_multiplier: Float[Array, " n_joints"]
    """Mimic multiplier for each joint. Shape: (n_joints,)."""
    mimic_offset: Float[Array, " n_joints"]
    """Mimic offset for each joint. Shape: (n_joints,)."""
    is_mimic: Bool[Array, " n_joints"]
    """Whether each joint is a mimic joint. Shape: (n_joints,)."""

    _topo_sort_inv: Int[Array, " n_joints"]
    """Inverse topological sort order, mapping sorted joint index to original joint index."""

    @jdc.jit
    def _map_to_full_joint_space(
        self,
        value_actuated: Float[Array, "*batch n_act_joints"],
        apply_offset: bool,
    ) -> Float[Array, "*batch n_joints"]:
        """Internal helper to map actuated values to full joint space,
        optionally applying mimic offset."""
        batch_axes = value_actuated.shape[:-1]
        assert value_actuated.shape == (*batch_axes, self.actuated_count)

        # Pad input vector to handle fixed joints (index -1) safely.
        value_padded = jnp.concatenate(
            [value_actuated, jnp.zeros((*batch_axes, 1))], axis=-1
        )
        safe_actuated_indices = jnp.where(
            self.actuated_indices == -1,
            self.actuated_count,  # Point to the zero padding
            self.actuated_indices,
        )

        # value_referenced contains the value of the joint each joint index refers to.
        value_referenced = value_padded[..., safe_actuated_indices]
        assert value_referenced.shape == (*batch_axes, self.count)

        # Apply mimic multiplier.
        value_multiplied = value_referenced * self.mimic_multiplier

        # Conditionally add mimic offset.
        value_full = jnp.where(
            apply_offset, value_multiplied + self.mimic_offset, value_multiplied
        )
        assert value_full.shape == (*batch_axes, self.count)

        return value_full

    @jdc.jit
    def get_full_config(
        self,
        cfg_actuated: Float[Array, "*batch n_act_joints"],
    ) -> Float[Array, "*batch n_joints"]:
        """Compute the full joint configuration vector (for all n_joints)
        from the configuration of the actuated joints (n_act_joints).

        Handles fixed joints and applies mimic joint relationships (multiplier + offset).

        Args:
            cfg_actuated: Configuration of the actuated joints.

        Returns:
            Full configuration vector.
        """
        return self._map_to_full_joint_space(cfg_actuated, apply_offset=True)

    @jdc.jit
    def get_full_derivative(
        self,
        derivative_actuated: Float[Array, "*batch n_act_joints"],
    ) -> Float[Array, "*batch n_joints"]:
        """Compute the full joint derivative vector (velocity, acceleration, etc.)
        for all n_joints from the derivative vector of the actuated joints (n_act_joints).

        Handles fixed joints (derivative=0) and applies mimic joint relationships (multiplier only).

        Args:
            derivative_actuated: Derivative vector (e.g., velocity) of the actuated joints.

        Returns:
            Full derivative vector.
        """
        return self._map_to_full_joint_space(derivative_actuated, apply_offset=False)


@jdc.pytree_dataclass
class LinkInfo:
    """Contains link-related information for a robot."""

    count: jdc.Static[int]
    names: jdc.Static[tuple[str, ...]]
    parent_joint_indices: Int[Array, " n_links"]


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
        num_joints = len(original_joints)
        original_name_to_idx = {j.name: i for i, j in enumerate(original_joints)}

        # Perform topological sort based on parent-child and mimic relationships.
        joints_to_sort = deepcopy(original_joints)
        sorted_joint_objects = list[yourdfpy.Joint]()
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

        if jnp.any(joint_order != jnp.arange(num_joints)):
            logger.info(
                "Joints were not in topological order; they will be internally sorted."
            )

        return joint_order

    @staticmethod
    def parse(urdf: yourdfpy.URDF) -> tuple[JointInfo, LinkInfo]:
        """Build joint and link information from a URDF."""
        joint_twists_list = list[Array]()
        parent_transform_list = list[Array]()
        parent_idx_list = list[int]()
        actuated_idx_list = list[int]()
        lower_limit_act_list = list[float]()
        upper_limit_act_list = list[float]()
        velocity_limit_act_list = list[float]()
        joint_name_list = list[str]()
        mimic_multiplier_list = list[float]()
        mimic_offset_list = list[float]()
        is_mimic_list = list[bool]()

        # Store limits read directly from URDF for *all* joints -> _eff
        lower_limit_eff_list = list[float]()
        upper_limit_eff_list = list[float]()
        velocity_limit_eff_list = list[float]()

        # Link information.
        link_name_list = list[str]()
        parent_joint_idx_list = list[int]()

        # First pass: collect joint information.
        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            # Get joint names.
            joint_name_list.append(joint.name)

            # Get twist for *this* joint (will be zero for fixed)
            twist = RobotURDFParser._get_joint_twist(joint)
            joint_twists_list.append(twist)

            # Get the actuated joint index it refers to (could be itself or mimicked).
            # Also get mimic parameters if applicable.
            act_idx, is_mimic, multiplier, offset = (
                RobotURDFParser._get_act_joint_idx_and_mimic(urdf, joint)
            )
            actuated_idx_list.append(act_idx)
            is_mimic_list.append(is_mimic)
            mimic_multiplier_list.append(multiplier)
            mimic_offset_list.append(offset)

            # Get effective joint limits.
            lower_eff, upper_eff = RobotURDFParser._get_joint_limits(joint)
            vel_limit_eff = RobotURDFParser._get_joint_limit_vel(joint)

            lower_limit_eff_list.append(lower_eff)
            upper_limit_eff_list.append(upper_eff)
            velocity_limit_eff_list.append(vel_limit_eff)

            # Get directly actuated joint limits.
            if joint in urdf.actuated_joints and not is_mimic:
                lower_limit_act_list.append(lower_eff)
                upper_limit_act_list.append(upper_eff)
                velocity_limit_act_list.append(vel_limit_eff)

            # Get the parent joint index and transform for each joint.
            parent_idx, T_parent_joint_val = RobotURDFParser._get_T_parent_joint(
                urdf, joint
            )
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

        # Convert collected lists to arrays
        lower_limits_act_arr = jnp.array(lower_limit_act_list)
        upper_limits_act_arr = jnp.array(upper_limit_act_list)
        velocity_limits_act_arr = jnp.array(velocity_limit_act_list)
        actuated_indices_arr = jnp.array(actuated_idx_list, dtype=jnp.int32)
        mimic_multiplier_arr = jnp.array(mimic_multiplier_list)
        mimic_offset_arr = jnp.array(mimic_offset_list)
        lower_limits_eff_arr = jnp.array(lower_limit_eff_list)
        upper_limits_eff_arr = jnp.array(upper_limit_eff_list)
        velocity_limits_eff_arr = jnp.array(velocity_limit_eff_list)

        # Create JointInfo and LinkInfo based on original order.
        joint_info = JointInfo(
            count=len(urdf.joint_map),
            actuated_count=len(urdf.actuated_joints),
            twists=jnp.array(joint_twists_list),
            parent_transforms=jnp.array(parent_transform_list),
            parent_indices=jnp.array(parent_idx_list, dtype=jnp.int32),
            actuated_indices=actuated_indices_arr,
            lower_limits_act=lower_limits_act_arr,
            upper_limits_act=upper_limits_act_arr,
            velocity_limits_act=velocity_limits_act_arr,
            names=tuple(joint_name_list),
            lower_limits_eff=lower_limits_eff_arr,
            upper_limits_eff=upper_limits_eff_arr,
            velocity_limits_eff=velocity_limits_eff_arr,
            mimic_multiplier=mimic_multiplier_arr,
            mimic_offset=mimic_offset_arr,
            is_mimic=jnp.array(is_mimic_list),
            _topo_sort_inv=topo_sort_inv_val,
        )
        assert joint_info.twists.shape == (joint_info.count, 6)
        assert joint_info.parent_transforms.shape == (joint_info.count, 7)
        assert joint_info.parent_indices.shape == (joint_info.count,)
        assert joint_info.actuated_indices.shape == (joint_info.count,)
        assert joint_info.lower_limits_act.shape == (joint_info.actuated_count,)
        assert joint_info.upper_limits_act.shape == (joint_info.actuated_count,)
        assert joint_info.velocity_limits_act.shape == (joint_info.actuated_count,)
        assert joint_info.lower_limits_eff.shape == (joint_info.count,)
        assert joint_info.upper_limits_eff.shape == (joint_info.count,)
        assert joint_info.velocity_limits_eff.shape == (joint_info.count,)
        assert joint_info._topo_sort_inv.shape == (joint_info.count,)
        assert joint_info.mimic_multiplier.shape == (joint_info.count,)
        assert joint_info.mimic_offset.shape == (joint_info.count,)
        assert joint_info.is_mimic.shape == (joint_info.count,)

        link_info = LinkInfo(
            count=len(link_name_list),
            names=tuple(link_name_list),
            parent_joint_indices=jnp.array(parent_joint_idx_list, dtype=jnp.int32),
        )
        assert link_info.parent_joint_indices.shape == (link_info.count,)
        return joint_info, link_info

    @staticmethod
    def _get_act_joint_idx_and_mimic(
        urdf: yourdfpy.URDF, joint: yourdfpy.Joint
    ) -> tuple[int, bool, float, float]:
        """Get the index of the actuated joint for a joint, and mimic parameters."""
        is_mimic = False
        multiplier = 1.0
        offset = 0.0

        # Check if this joint is a mimic joint.
        if joint.mimic is not None:
            is_mimic = True
            if joint.mimic.multiplier is not None:
                multiplier = joint.mimic.multiplier
            if joint.mimic.offset is not None:
                offset = joint.mimic.offset

            mimicked_joint_name = joint.mimic.joint
            mimicked_joint = urdf.joint_map[mimicked_joint_name]
            act_joint_idx = urdf.actuated_joints.index(mimicked_joint)

        # If not mimic, check if it's directly actuated.
        elif joint in urdf.actuated_joints:
            assert joint.axis.shape == (3,)
            act_joint_idx = urdf.actuated_joints.index(joint)

        # Otherwise, it's a fixed joint.
        else:
            act_joint_idx = -1  # Represents non-actuated/fixed.

        return act_joint_idx, is_mimic, multiplier, offset

    @staticmethod
    def _get_joint_twist(joint: yourdfpy.Joint) -> Array:
        """Get the twist parameters for any joint (zero for fixed)."""
        if joint.type in ("revolute", "continuous"):
            twist = jnp.concatenate([jnp.zeros(3), joint.axis])
        elif joint.type == "prismatic":
            twist = jnp.concatenate([joint.axis, jnp.zeros(3)])
        elif joint.type == "fixed":
            twist = jnp.zeros(6)
        else:
            # Floating joints etc. are not supported yet.
            logger.warning(
                f"Unsupported joint type {joint.type} encountered for joint '{joint.name}'. Treating as fixed."
            )
            twist = jnp.zeros(6)
            # raise ValueError(f"Unsupported joint type {joint.type}!")
        assert twist.shape == (6,)
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
        """Get the joint limits defined in the URDF.
        Assumes caller checked that joint.limit.lower/upper exist OR type is fixed/continuous.
        """
        if joint.type == "fixed":
            return 0.0, 0.0
        elif joint.type == "continuous":
            if (
                joint.limit is not None
                and joint.limit.lower is not None
                and joint.limit.upper is not None
            ):
                return joint.limit.lower, joint.limit.upper
            else:
                logger.warning(
                    f"Continuous joint '{joint.name}' has no explicit limits."
                    "Returning [-pi, pi]."
                )
                return -jnp.pi, jnp.pi
        elif (
            joint.limit is not None
            and joint.limit.lower is not None
            and joint.limit.upper is not None
        ):
            return joint.limit.lower, joint.limit.upper
        elif joint.type == "fixed":
            return 0.0, 0.0
        else:
            raise ValueError(f"Joint '{joint.name}' ({joint.type}) has no limits.")

    @staticmethod
    def _get_joint_limit_vel(joint: yourdfpy.Joint) -> float:
        """Get the joint velocity limit defined in the URDF.
        Assumes checks for existence have been done prior to calling, or joint is fixed.
        """
        if joint.type == "fixed":
            return 0.0
        elif joint.limit is not None and joint.limit.velocity is not None:
            return joint.limit.velocity
        else:
            raise ValueError(
                f"Joint '{joint.name}' of type '{joint.type}' has no velocity limits."
            )
