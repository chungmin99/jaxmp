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

    joint: JointInfo
    """Joint information for the robot."""

    link: LinkInfo
    """Link information for the robot."""

    unroll_fk: jdc.Static[bool]
    """Whether to unroll the forward kinematics `fori_loop`."""

    JointVar: jdc.Static[type[jaxls.Var[Array]]]
    """Variable class for the robot configuration."""

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
        joint, link = RobotURDFParser.parse(urdf)

        # Use actuated limits for default value and retract scaling
        default_val = (joint.lower_limits_act + joint.upper_limits_act) / 2
        JointVar = Robot.get_joint_var_class(
            default_val=default_val,
            num_actuated_joints=joint.actuated_count,
            joint_vel_limit=joint.velocity_limits_act,
        )

        robot = Robot(
            joint=joint,
            link=link,
            unroll_fk=unroll_fk,
            JointVar=JointVar,
        )

        return robot

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch count 7"]:
        """
        Run forward kinematics on the robot, in the provided configuration.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch actuated_count)`.

        Returns:
            The SE(3) transforms of the joints, in the format `(*batch count wxyz_xyz)`.
        """
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.joint.actuated_count)

        # Calculate full configuration using the dedicated method
        q_full = self.joint.get_full_config(cfg)

        # Calculate delta transforms using the effective config and twists for all joints.
        tangents = self.joint.twists * q_full[..., None]
        assert tangents.shape == (*batch_axes, self.joint.count, 6)
        delta_Ts = jaxlie.SE3.exp(tangents)  # Shape: (*batch_axes, self.joint.count, 7)

        # Combine constant parent transform with variable joint delta transform.
        Ts_parent_child = (jaxlie.SE3(self.joint.parent_transforms) @ delta_Ts).wxyz_xyz
        assert Ts_parent_child.shape == (*batch_axes, self.joint.count, 7)

        # In this function we leverage two index mappings:
        # 1. sort_order: map original_idx -> sorted_idx.
        # 2. self.joint.topo_sort_inv: map sorted_idx -> original_idx.
        # We use these mappings to convert between the original and topologically sorted orderings.

        # 1. Calculate topological sort order (original -> sorted).
        topo_order = jnp.argsort(self.joint._topo_sort_inv)

        # 2. Convert Ts_parent_child to topologically sorted order.
        # This is slightly counterintuitive:
        #   output[..., i, :] gets populated with the value from input[..., self.joint.topo_sort_inv[i], :].
        #   Since self.joint.topo_sort_inv[i] gives the original index for sorted index i,
        #   this correctly gathers the transforms from their original positions into the new sorted order.
        Ts_parent_child_sorted = Ts_parent_child[..., self.joint._topo_sort_inv, :]

        # 3. Calculate parent's original_idx for each child's sorted_idx.
        parent_orig_for_sorted_child = self.joint.parent_indices[
            self.joint._topo_sort_inv
        ]

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
                    jaxlie.SE3(T_world_parent)
                    @ jaxlie.SE3(Ts_parent_child_sorted[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_joint_init_sorted = jnp.zeros((*batch_axes, self.joint.count, 7))
        Ts_world_joint_sorted = jax.lax.fori_loop(
            lower=0,
            upper=self.joint.count,
            body_fun=compute_joint,
            init_val=Ts_world_joint_init_sorted,
            unroll=self.unroll_fk,
        )

        # 6. Gather elements into original order using sort_order (original_idx -> sorted_idx).
        Ts_world_joint = Ts_world_joint_sorted[..., topo_order, :]
        assert Ts_world_joint.shape == (
            *batch_axes,
            self.joint.count,
            7,
        )

        return Ts_world_joint

    @jdc.jit
    def forward_kinematics_links(
        self,
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch count 7"]:
        """Run forward kinematics on the robot's links, in the provided configuration.

        Returns transforms in the order corresponding to `self.link.names`.
        """
        raise NotImplementedError

    @staticmethod
    def get_joint_var_class(
        default_val: Float[Array, "* actuated_count"],
        num_actuated_joints: int,
        joint_vel_limit: Float[Array, " actuated_count"],
    ) -> type[jaxls.Var[Array]]:
        """Return a variable class for the robot configuration,
        considering different joint units for revolute/prismatic joints."""

        @jdc.jit
        def retract_fn(
            cfg: Float[Array, "*batch actuated_count"],
            delta: Float[Array, "*batch actuated_count"],
        ) -> Float[Array, "*batch actuated_count"]:
            """Retract function for the robot."""
            assert cfg.shape == delta.shape
            assert cfg.shape[-1] == num_actuated_joints

            # Apply units to delta, by normalizing w/ the joint velocity.
            # Uses the velocity limits of the *actuated* joints.
            _delta = delta * joint_vel_limit * 0.01

            return cfg + _delta

        class JointVar(  # pylint: disable=missing-class-docstring
            jaxls.Var[Array],
            default_factory=lambda: default_val.copy(),
            tangent_dim=num_actuated_joints,
            retract_fn=retract_fn,
        ): ...

        return JointVar
