from __future__ import annotations

import jax
import jax_dataclasses as jdc
import jaxlie
import yourdfpy

from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int

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

    _joint_sort_order_inv: Int[Array, " joints"]
    """Inverse topological sort order for the joints (maps sorted idx -> original idx)."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        unroll_fk: bool = False,
    ) -> Robot:
        """
        Loads a robot kinematic tree from a URDF.
        Internally tracks a topological sort of the joints.

        Args:
            urdf: The URDF to load the robot from.
            unroll_fk: Whether to unroll the forward kinematics loop (`fori_loop`).
        """
        # 1. Parse URDF in original order.
        joint_info_orig, link_info_orig = RobotURDFParser.parse(urdf)

        # 2. Get topological sort information.
        joint_sort_order_inv = RobotURDFParser._topologically_sort_joints(urdf)

        # 3. Create Robot instance.
        default_val = (joint_info_orig.limits_lower + joint_info_orig.limits_upper) / 2

        JointVar = Robot.get_joint_var_class(
            default_val=default_val,
            num_actuated_joints=joint_info_orig.num_actuated_joints,
            joint_vel_limit=joint_info_orig.joint_vel_limit,
        )

        robot = Robot(
            joint_info=joint_info_orig,
            link_info=link_info_orig,
            unroll_fk=unroll_fk,
            JointVar=JointVar,
            _joint_sort_order_inv=joint_sort_order_inv,
        )

        return robot

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
            jaxlie.SE3(self.joint_info.Ts_parent_joint)
            @ jaxlie.SE3(Ts_joint_child)
        ).wxyz_xyz

        # In this function we leverage two index mappings:
        # 1. sort_order: map original_idx -> sorted_idx.
        # 2. self._joint_sort_order_inv: map sorted_idx -> original_idx.
        # We use these mappings to convert between the original and topologically sorted orderings.

        # 1. Calculate topological sort order (original -> sorted).
        topo_order = jnp.argsort(self._joint_sort_order_inv)

        # 2. Convert Ts_parent_child to topologically sorted order.
        # This is slightly counterintuitive:
        #   output[..., i, :] gets populated with the value from input[..., _joint_sort_order_inv[i], :].
        #   Since _joint_sort_order_inv[i] gives the original index for sorted index i,
        #   this correctly gathers the transforms from their original positions into the new sorted order.
        Ts_parent_child_sorted = Ts_parent_child[..., self._joint_sort_order_inv, :]

        # 3. Calculate parent's original_idx for each child's sorted_idx.
        parent_orig_for_sorted_child = self.joint_info.idx_parent_joint[self._joint_sort_order_inv]

        # 4. Calculate parent's sorted_idx for each child's sorted_idx.
        idx_parent_joint_sorted = jnp.where(
            parent_orig_for_sorted_child == -1,
            -1,
            topo_order[parent_orig_for_sorted_child],
        )

        # 5. Compute transforms, within topologically sorted order.
        def compute_joint(i: int, Ts_world_joint_sorted: Array) -> Array:
            parent_sorted_idx = idx_parent_joint_sorted[i]
            T_world_parent = jnp.where(
                parent_sorted_idx == -1,
                jaxlie.SE3.identity().wxyz_xyz,
                Ts_world_joint_sorted[..., parent_sorted_idx, :],
            )
            return Ts_world_joint_sorted.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent) @ jaxlie.SE3(Ts_parent_child_sorted[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_joint_init_sorted = jnp.zeros((*batch_axes, self.joint_info.num_joints, 7))
        Ts_world_joint_sorted = jax.lax.fori_loop(
            lower=0,
            upper=self.joint_info.num_joints,
            body_fun=compute_joint,
            init_val=Ts_world_joint_init_sorted,
            unroll=self.unroll_fk,
        )

        # 6. Gather elements into original order using sort_order (original_idx -> sorted_idx).
        Ts_world_joint = Ts_world_joint_sorted[..., topo_order, :]
        assert Ts_world_joint.shape == (
            *batch_axes,
            self.joint_info.num_joints,
            7,
        )

        return Ts_world_joint

    @jdc.jit
    def forward_kinematics_links(
        self,
        cfg: Float[Array, "*batch num_act_joints"],
    ) -> Float[Array, "*batch num_links 7"]:
        """Run forward kinematics on the robot's links, in the provided configuration.

        Returns transforms in the order corresponding to `self.link_info.link_names`.
        """
        raise NotImplementedError

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
