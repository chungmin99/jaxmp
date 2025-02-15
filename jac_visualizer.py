from __future__ import annotations

import time
from typing import Literal

import numpy as np
import tyro

import jax
import jax.numpy as jnp
import jaxlie

import viser
from viser.extras import ViserUrdf

from jaxmp import JaxKinTree
from jaxmp.extras.urdf_loader import load_urdf

from typing import Literal

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import jax_dataclasses as jdc

from jaxmp.robot_factors import RobotFactors
from jaxmp.kinematics import JaxKinTree
from jaxmp.coll import RobotColl, CollGeom
from jaxmp.extras import solve_ik


@jdc.jit
def solve_ik_custom(
    kin: JaxKinTree,
    target_pose: jaxlie.SE3,
    target_joint_indices: jax.Array,
    initial_pose: jnp.ndarray,
    JointVar: jdc.Static[type[jaxls.Var[jax.Array]]],
    ik_weight: jnp.ndarray,
    *,
    joint_var_idx: int = 0,
    rest_weight: float = 0.001,
    limit_weight: float = 100.0,
    joint_vel_weight: float = 0.0,
    dt: float = 0.01,
    use_manipulability: jdc.Static[bool] = False,
    manipulability_weight: float = 0.001,
    solver_type: jdc.Static[
        Literal["cholmod", "conjugate_gradient", "dense_cholesky"]
    ] = "conjugate_gradient",
    ConstrainedSE3Var: jdc.Static[type[jaxls.Var[jaxlie.SE3]] | None] = None,
    pose_var_idx: int = 0,
    max_iterations: int = 50,
) -> jnp.ndarray:
    """
    Solve IK for the robot.
    Args:
        target_pose: Desired pose of the target joint, SE3 has batch axes (n_target,).
        target_joint_indices: Indices of the target joints, length n_target.
        initial_pose: Initial pose of the joints, used for joint velocity cost factor.
        JointVar: Joint variable type.
        ConstrainedSE3Var: Constrained SE3 variable type.
        joint_var_idx: Index for the joint variable.
        pose_var_idx: Index for the pose variable.
        ik_weight: Weight for the IK cost factor.
        rest_weight: Weight for the rest cost factor.
        limit_weight: Weight for the joint limit cost factor.
        joint_vel_weight: Weight for the joint velocity cost factor.
        solver_type: Type of solver to use.
        dt: Time step for the velocity cost factor.
        max_iterations: Maximum number of iterations for the solver.
        manipulability_weight: Weight for the manipulability cost factor.
    Returns:
        Base pose (jaxlie.SE3)
        Joint angles (jnp.ndarray)
    """
    # NOTE You can't add new factors on-the-fly with JIT, because:
    # - we'd want to pass in lists of jaxls.Factor objects
    # - but lists / tuples are static
    # - and ArrayImpl is not a valid type for a static argument.
    # (and you can't stack different Factor definitions, since it's a part of the treedef.)

    factors: list[jaxls.Factor] = [
        RobotFactors.limit_cost_factor(
            JointVar,
            joint_var_idx,
            kin,
            jnp.array([limit_weight] * kin.num_actuated_joints),
        ),
        RobotFactors.limit_vel_cost_factor(
            JointVar,
            joint_var_idx,
            kin,
            dt,
            jnp.array([joint_vel_weight] * kin.num_actuated_joints),
            prev_cfg=initial_pose,
        ),
        RobotFactors.rest_cost_factor(
            JointVar,
            joint_var_idx,
            jnp.array([rest_weight] * kin.num_actuated_joints),
        ),
    ]

    def ik_cost(
        vals: jaxls.VarValues,
        var: jaxls.Var[jax.Array],
        target_joint_idx: jax.Array | int,
        weights: jnp.ndarray,
    ):
        # Handle world-to-base transform.
        joint_cfg: jax.Array = vals[var]
        Ts_joint_world = kin.forward_kinematics(joint_cfg)
        residual = jaxlie.SE3(Ts_joint_world[target_joint_idx]).translation() - target_pose.translation()
        # residual = (
        #     (jaxlie.SE3(Ts_joint_world[target_joint_idx])).inverse() @ (target_pose)
        # ).translation()
        return (residual * weights).flatten()

    factors.append(
        jaxls.Factor(
            ik_cost,
            (
                JointVar(joint_var_idx),
                target_joint_indices,
                ik_weight,
            ),
            "custom",
            jac_custom_fn=lambda vals, var, target_joint_idx, weights: jac_forward(
                kin, vals[var], target_joint_idx
            ),
        ),
    )

    if use_manipulability:
        factors.append(
            RobotFactors.manipulability_cost_factor(
                JointVar,
                joint_var_idx,
                kin,
                target_joint_indices,
                manipulability_weight,
            )
        )

    joint_vars: list[jaxls.Var] = [JointVar(joint_var_idx)]
    joint_var_values: list[jaxls.Var | jaxls._variables.VarWithValue] = [
        JointVar(joint_var_idx).with_value(initial_pose)
    ]
    if ConstrainedSE3Var is not None and pose_var_idx is not None:
        joint_vars.append(ConstrainedSE3Var(pose_var_idx))
        joint_var_values.append(ConstrainedSE3Var(pose_var_idx))

    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars,
        use_onp=False,
    )
    solution = graph.solve(
        linear_solver=solver_type,
        initial_vals=jaxls.VarValues.make(joint_var_values),
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5,
            parameter_tolerance=1e-5,
            max_iterations=max_iterations,
        ),
        verbose=False,
    )

    if ConstrainedSE3Var is not None:
        base_pose = solution[ConstrainedSE3Var(0)]
    else:
        base_pose = jaxlie.SE3.identity()

    joints = solution[JointVar(0)]
    return joints


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < 0 and upper > 0 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


# def hat(x: jnp.ndarray) -> jnp.ndarray:
#     return jnp.array(
#         [
#             [0, -x[2], x[1]],
#             [x[2], 0, -x[0]],
#             [-x[1], x[0], 0],
#         ]
#     )

# def jac_forward(kin: JaxKinTree, cfg: jnp.ndarray) -> jnp.ndarray:
#     # jac: (num_joints, 7, num_actuated_joints)
#     Ts_world_joint = kin.forward_kinematics(cfg)
#     Ts_parent_joint = kin.Ts_parent_joint
#     jac = jnp.zeros((kin.num_joints, 7, kin.num_actuated_joints))
#     # I know that all of them are revolute joints.
#     for i in range(kin.num_joints):
#         for j in range(kin.num_actuated_joints):
#             T_parent_joint = jnp.where(
#                 kin.idx_parent_joint[i] == -1,
#                 jaxlie.SE3.identity().wxyz_xyz,
#                 jaxlie.SE3(Ts_world_joint[kin.idx_parent_joint[i]]).wxyz_xyz,
#             )
#             T_parent_joint = jaxlie.SE3(T_parent_joint) # @ jaxlie.SE3(Ts_parent_joint[i])
#             # upper = T_parent_joint.rotation() @ jaxlie.SO3.from_matrix(
#             #     hat(kin.joint_twists[j, 3:])
#             # ) # <-- this is wrong
#             # jac = jac.at[i, :4, j].set(
#             #     jnp.where(
#             #         j <= kin.idx_actuated_joint[i],
#             #         upper.wxyz,
#             #         jnp.zeros(4),
#             #     )
#             # )
#             # jac = jac.at[i, 4:, j].set(T_parent_joint.translation())
#     return jac

# def hat(vec):
#     """Convert a 3D vector to a skew-symmetric matrix."""
#     x, y, z = vec
#     return jnp.array([
#         [0, -z, y],
#         [z, 0, -x],
#         [-y, x, 0]
#     ])

# def adjoint(T: jaxlie.SE3):
#     """Compute the adjoint matrix for SE(3)."""
#     R = T.rotation().as_matrix()
#     p = T.translation()
#     p_hat = hat(p)
#     upper = jnp.block([[R, jnp.zeros((3, 3))],
#                        [p_hat @ R, R]])
#     return upper

# def jac_forward(kin: JaxKinTree, cfg: jnp.ndarray) -> jnp.ndarray:
#     Ts_world_joint = kin.forward_kinematics(cfg)
#     jac = jnp.zeros((kin.num_joints, 7, kin.num_actuated_joints))

#     for i in range(kin.num_joints):
#         joint_pose = jaxlie.SE3(Ts_world_joint[i])

#         # Compute Adjoint for the current joint's world transform
#         Ad_T = adjoint(joint_pose)

#         for j in range(kin.num_actuated_joints):
#             twist = kin.joint_twists[j]  # Local twist: [v; ω]

#             # Transform twist to world frame
#             world_twist = Ad_T @ twist

#             # Quaternion derivative (wxyz)
#             omega = world_twist[3:]
#             w, x, y, z = joint_pose.rotation().wxyz
#             wx, wy, wz = omega

#             q_dot = 0.5 * jnp.array([
#                 -x * wx - y * wy - z * wz,
#                 w * wx + y * wz - z * wy,
#                 w * wy - x * wz + z * wx,
#                 w * wz + x * wy - y * wx
#             ])

#             # Linear velocity: world_twist[:3]
#             linear_velocity = world_twist[:3]

#             # Populate the Jacobian
#             if j <= kin.idx_actuated_joint[i]:
#                 jac = jac.at[i, :4, j].set(q_dot)
#                 jac = jac.at[i, 4:, j].set(linear_velocity)

#     return jac


def jac_forward(
    kin: JaxKinTree, cfg: jnp.ndarray, target_joint_idx: int
) -> jnp.ndarray:
    """Forward kinematics Jacobian."""
    Ts_world_joint = kin.forward_kinematics(cfg)

    T_target_joint = jaxlie.SE3(Ts_world_joint[target_joint_idx])

    # We need to get the exact indices of the actuated joints ! including the mimic joints
    # Like the exact kinematic chain.

    # For the mimic joints, maybe I can sum them.
    def body_fun(i, curr_parent, idx):
        act_joint_idx = -1
        act_joint_idx = jnp.where(
            i == curr_parent,
            kin.idx_actuated_joint[i],
            -1,
        )

        curr_parent = jnp.where(
            i == curr_parent,
            kin.idx_parent_joint[i],
            curr_parent,
        )

        return (
            i - 1,
            curr_parent,
            idx.at[i].set(
                jnp.where(
                    i > target_joint_idx,
                    -1,
                    act_joint_idx,
                )
            ),
        )

    idx_applied_to_target = jnp.zeros(kin.num_joints, dtype=jnp.int32)
    idx_applied_to_target = jax.lax.while_loop(
        lambda carry: (carry[0] >= 0),
        lambda carry: body_fun(*carry),
        (kin.num_joints - 1, target_joint_idx, idx_applied_to_target),
    )[-1]

    # Get the kinematic chain
    Ts_world_act_joint = jaxlie.SE3(Ts_world_joint)
    joint_twists = kin.joint_twists[kin.idx_actuated_joint]
    vel = joint_twists[..., :3]
    omega = joint_twists[..., 3:]

    parent_translation = T_target_joint.translation() - Ts_world_act_joint.translation()
    omega = Ts_world_act_joint.rotation() @ omega
    vel = Ts_world_act_joint.rotation() @ vel

    linear_part = jnp.cross(omega, b=parent_translation).squeeze() + vel.squeeze()

    # Gather the linear part into num_actuated_joints, using the idx_applied_to_target
    # breakpoint()
    jac = jnp.zeros((3, kin.num_actuated_joints))
    jac = jac.at[:, idx_applied_to_target].add(
        jnp.where(
            (idx_applied_to_target != -1)[None],
            linear_part.T,
            jnp.zeros((3, 1)),
        )
    )
    # jac = jac.at[:4, idx_applied_to_target].add(
    #     jnp.where(
    #         (idx_applied_to_target != -1)[None],
    #         rot_part.T,
    #         jnp.zeros((4, 1)),
    #     )
    # )

    assert jac.shape == (3, kin.num_actuated_joints)
    return jac

    ###

    Ts_world_act_joint = jaxlie.SE3(Ts_world_joint[: kin.num_actuated_joints])

    omega = kin.joint_twists[..., 3:]

    parent_translation = T_target_joint.translation() - Ts_world_act_joint.translation()
    omega = Ts_world_act_joint.rotation() @ omega
    linear_part = jnp.cross(omega, b=parent_translation).squeeze()

    jac = linear_part.T
    assert jac.shape == (3, kin.num_actuated_joints)
    return jac

    # Ts

    # Assume all revolute joints.
    # dep_idx = kin.idx_parent_joint[target_joint_idx]
    # joint_translation = Ts_world_joint[target_joint_idx][..., 4:]

    # for j in range(kin.num_joints - 1, -1, -1):
    #     # Extract the joint twist
    #     act_j = kin.idx_actuated_joint[j]
    #     twist = kin.joint_twists[act_j]

    #     omega = twist[3:]

    #     # Compute linear part: omega x (joint_translation - parent_translation)
    #     T_parent = jaxlie.SE3(Ts_world_joint[j])
    #     parent_translation = jnp.where(
    #         kin.idx_parent_joint[j] == -1, jnp.zeros(3), T_parent.translation()
    #     )
    #     r = joint_translation - parent_translation

    #     omega = T_parent.rotation().as_matrix() @ omega
    #     linear_part = jnp.cross(omega, r).squeeze()
    #     angular_part = omega.squeeze()

    #     # Fill in the Jacobian
    #     jac = jac.at[:3, act_j].add(
    #         jnp.where(j == dep_idx, linear_part.squeeze(), jnp.zeros(3))
    #     )
    #     jac = jac.at[3:, act_j].add(
    #         jnp.where(j == dep_idx, angular_part.squeeze(), jnp.zeros(3))
    #     )

    #     dep_idx = jnp.where(
    #         dep_idx == j,
    #         kin.idx_parent_joint[dep_idx],
    #         dep_idx,
    #     )

    # adjoint = jaxlie.SE3(Ts_world_joint[target_joint_idx]).inverse().adjoint().squeeze()
    # jac = adjoint @ jac
    # return jac


def main(
    robot_type: Literal[
        "panda",
        "ur10",
        "cassie",
        "allegro_hand",
        "barrett_hand",
        "robotiq_2f85",
        "atlas_drc",
        "g1",
        "h1",
        "anymal_c",
        "go2",
        "yumi",
        "ur5",
    ] = "panda",
) -> None:
    # Start viser server.
    server = viser.ViserServer(port=8081)

    # Load URDF.
    #
    # This takes either a yourdfpy.URDF object or a path to a .urdf file.
    urdf = load_urdf(robot_type + "_description")
    viser_urdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/urdf")
    kin = JaxKinTree.from_urdf(urdf)

    rest_pose = (kin.limits_lower + kin.limits_upper) / 2.0
    # target_joint_idx = urdf.joint_names.index("panda_joint8")
    # target_joint_idx = urdf.joint_names.index("panda_hand_tcp_joint")
    # breakpoint()

    cfg = rest_pose
    for target_joint_idx in range(kin.num_joints):
        print(f"target_joint_idx: {target_joint_idx}")
        gt_jac_fn = jax.jacfwd(
            lambda cfg: jaxlie.SE3(
                kin.forward_kinematics(cfg)[target_joint_idx]
            ).translation()
        )
        gt_jac = gt_jac_fn(cfg)
        print(f"gt_jac: \n{gt_jac.T.round(2)}")

        our_jac = jac_forward(kin, cfg, target_joint_idx)
        print(f"our_jac: \n{our_jac.T.round(2)}")
        assert jnp.allclose(gt_jac, our_jac, atol=1e-3)

    breakpoint()

    # Create sliders in GUI that help us move the robot joints.
    # with server.gui.add_folder("Joint position control"):
    #     (slider_handles, initial_config) = create_robot_control_sliders(
    #         server, viser_urdf
    #     )

    # Set initial robot configuration.
    initial_config = rest_pose
    viser_urdf.update_cfg(np.array(initial_config))

    # Create joint reset button.
    # reset_button = server.gui.add_button("Reset")

    # @reset_button.on_click
    # def _(_):
    #     for s, init_q in zip(slider_handles, initial_config):
    #         s.value = init_q

    def draw_line(name, start, end, color=(0, 0, 0)):
        positions = np.stack([start, end], axis=0)
        server.scene.add_spline_cubic_bezier(
            name, positions, positions, line_width=2, color=color
        )

    # def update_jac():
    #     current_cfg = jnp.array([slider.value for slider in slider_handles])

    #     Ts_world_joint = kin.forward_kinematics(current_cfg)
    #     gt_jac_fn = jax.jacfwd(
    #         lambda cfg: jaxlie.SE3(
    #             kin.forward_kinematics(cfg)[target_joint_idx]
    #         ).translation()
    #     )
    #     jac = gt_jac_fn(current_cfg)
    #     jac_ours = jac_forward(kin, current_cfg, target_joint_idx)

    #     assert Ts_world_joint.shape == (kin.num_joints, 7)
    #     assert jac.shape == (3, kin.num_actuated_joints)

    #     start = Ts_world_joint[target_joint_idx, 4:]
    #     for j in range(kin.num_actuated_joints):
    #         end = start + jac[:, j]
    #         draw_line(f"jac/jac_joint_{j}", np.array(start), np.array(end))

    #         end = start + jac_ours[:, j]
    #         draw_line(
    #             f"jac_ours/jac_joint_{j}",
    #             np.array(start),
    #             np.array(end),
    #             color=(255, 0, 0),
    #         )

    # Add target transform controls
    target_tf = server.scene.add_transform_controls(
        "target_transform", scale=0.2
    )
    target_frame = server.scene.add_frame(
        "target",
        axes_length=0.1,
        axes_radius=0.01,
        origin_radius=0.02,
    )

    # Add GUI elements for timing and joint selection
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    target_joint_handle = server.gui.add_dropdown(
        "Target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[-1],  # Default to last joint
    )

    # Add smooth IK option
    smooth_handle = server.gui.add_checkbox("Smooth", initial_value=False)

    # Set initial target position to selected joint
    # current_cfg = jnp.array([slider.value for slider in slider_handles])
    Ts_world_joint = kin.forward_kinematics(initial_config)
    target_joint_idx = urdf.joint_names.index(target_joint_handle.value)
    T_target = jaxlie.SE3(Ts_world_joint[target_joint_idx])
    
    target_tf.position = np.array(T_target.translation())
    target_tf.wxyz = np.array(T_target.rotation().wxyz)
    target_frame.position = np.array(T_target.translation())
    target_frame.wxyz = np.array(T_target.rotation().wxyz)

    # Add button to set target to current pose
    set_target_to_current = server.gui.add_button("Set target to current pose")

    current_cfg = initial_config
    @set_target_to_current.on_click
    def _(_):
        # current_cfg = jnp.array([slider.value for slider in slider_handles])
        Ts_world_joint = kin.forward_kinematics(current_cfg)
        target_joint_idx = urdf.joint_names.index(target_joint_handle.value)
        T_target = jaxlie.SE3(Ts_world_joint[target_joint_idx])
        
        target_tf.position = np.array(T_target.translation())
        target_tf.wxyz = np.array(T_target.rotation().wxyz)
        target_frame.position = np.array(T_target.translation())
        target_frame.wxyz = np.array(T_target.rotation().wxyz)

    has_jitted = False
    # Sleep forever.
    JointVar = RobotFactors.get_var_class(kin, rest_pose)
    while True:
        # current_cfg = jnp.array([slider.value for slider in slider_handles])
        target_pose = jaxlie.SE3(
            jnp.array([*target_tf.wxyz, *target_tf.position])
        )
        target_joint_idx = urdf.joint_names.index(target_joint_handle.value)

        # Set IK parameters based on smooth mode
        if smooth_handle.value:
            initial_pose = current_cfg
            joint_vel_weight = 100.0  # Same as limit_weight
        else:
            initial_pose = current_cfg  # Use current instead of rest to avoid jumps
            joint_vel_weight = 0.0

        # Solve IK with timing
        start_time = time.time()
        new_joints = solve_ik_custom(
            kin,
            target_pose=target_pose,
            target_joint_indices=jnp.array(target_joint_idx),
            initial_pose=initial_pose,
            JointVar=JointVar,
            ik_weight=jnp.array([5.0, 5.0, 5.0]),  # Position weights
            rest_weight=0.01,
            limit_weight=100.0,
            joint_vel_weight=joint_vel_weight,
        )

        # Ensure computation is complete before timing
        jax.block_until_ready(new_joints)
        timing_handle.value = (time.time() - start_time) * 1000
        if not has_jitted:
            print(f"JIT compile + running took {timing_handle.value:.1f} ms")
            has_jitted = True

        # Update robot configuration
        # for slider, joint_val in zip(slider_handles, new_joints):
        #     slider.value = float(joint_val)
        current_cfg = new_joints
        viser_urdf.update_cfg(np.array(current_cfg))

        # Update jacobian visualization
        # update_jac()
        # time.sleep(0.1)


if __name__ == "__main__":
    tyro.cli(main)
