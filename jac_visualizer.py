from __future__ import annotations

import time
from typing import Literal

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import viser
from loguru import logger
from tyro.extras import SubcommandApp
from viser.extras import ViserUrdf

from jaxmp import JaxKinTree
from jaxmp.extras.urdf_loader import load_urdf
from jaxmp.kinematics import JaxKinTree
from jaxmp.robot_factors import RobotFactors

app = SubcommandApp()


def _skew(omega: jax.Array) -> jax.Array:
    """Returns the skew-symmetric form of a length-3 vector."""

    wx, wy, wz = jnp.moveaxis(omega, -1, 0)
    zeros = jnp.zeros_like(wx)
    return jnp.stack(
        [zeros, -wz, wy, wz, zeros, -wx, -wy, wx, zeros],
        axis=-1,
    ).reshape((*omega.shape[:-1], 3, 3))


def V_inv(theta: jax.Array) -> jax.Array:
    theta_squared = jnp.sum(jnp.square(theta), axis=-1)
    use_taylor = theta_squared < 1e-5

    # Shim to avoid NaNs in jnp.where branches, which cause failures for
    # reverse-mode AD.
    theta_squared_safe = jnp.where(
        use_taylor,
        jnp.ones_like(theta_squared),  # Any non-zero value should do here.
        theta_squared,
    )
    del theta_squared
    theta_safe = jnp.sqrt(theta_squared_safe)
    half_theta_safe = theta_safe / 2.0

    skew_omega = _skew(theta)
    V_inv = jnp.where(
        use_taylor[..., None, None],
        jnp.eye(3)
        - 0.5 * skew_omega
        + jnp.einsum("...ij,...jk->...ik", skew_omega, skew_omega) / 12.0,
        (
            jnp.eye(3)
            - 0.5 * skew_omega
            + (
                (
                    1.0
                    - theta_safe
                    * jnp.cos(half_theta_safe)
                    / (2.0 * jnp.sin(half_theta_safe))
                )
                / theta_squared_safe
            )[..., None, None]
            * jnp.einsum("...ij,...jk->...ik", skew_omega, skew_omega)
        ),
    )
    return V_inv


def jac_position_and_orientation_cost(
    kin: JaxKinTree,
    cfg: jnp.ndarray,
    target_joint_idx: int,
    idx_applied_to_target: jax.Array,
    target: jaxlie.SE3,
) -> jnp.ndarray:
    """Jacobian for doing basic IK."""
    Ts_world_joint = kin.forward_kinematics(cfg)
    T_world_target = jaxlie.SE3(Ts_world_joint[target_joint_idx])

    # Get the kinematic chain
    Ts_world_act_joint = jaxlie.SE3(Ts_world_joint)
    joint_twists = kin.joint_twists[kin.idx_actuated_joint]
    vel = joint_twists[..., :3]
    omega = joint_twists[..., 3:]

    ### Get the translational component.
    # Get the kinematic chain
    parent_translation = T_world_target.translation() - Ts_world_act_joint.translation()
    omega = Ts_world_act_joint.rotation() @ omega
    vel = Ts_world_act_joint.rotation() @ vel

    linear_part = jnp.cross(omega, b=parent_translation).squeeze() + vel.squeeze()
    jac_translation = jnp.zeros((3, kin.num_actuated_joints))
    jac_translation = jac_translation.at[:, idx_applied_to_target].add(
        jnp.where(
            (idx_applied_to_target != -1)[None],
            linear_part.T,
            jnp.zeros((3, 1)),
        )
    )
    assert jac_translation.shape == (3, kin.num_actuated_joints)

    omega_wrt_ee = T_world_target.rotation().inverse() @ omega
    ori_error = T_world_target.rotation().inverse() @ target.rotation()
    Jlog3 = V_inv(ori_error.log())

    jac_orientation = jnp.zeros((3, kin.num_actuated_joints))
    jac_orientation = jac_orientation.at[:, idx_applied_to_target].add(
        jnp.where(
            (idx_applied_to_target != -1)[None],
            Jlog3 @ omega_wrt_ee.T,
            jnp.zeros((3, 1)),
        )
    )
    # need negative because inverse is on target term, I think
    jac_orientation = -jac_orientation

    return (
        jnp.concatenate(
            [
                jac_translation,
                jac_orientation,
            ],
            axis=0,
        )
        # (1, num_actuated_joints)
        * kin._get_tangent_scaling_terms()[None, :]
    )


def position_and_orientation_cost(
    kin: JaxKinTree,
    cfg: jax.Array,
    target: jaxlie.SE3,
    target_joint_idx: int | jax.Array,
) -> jnp.ndarray:
    T_world_target = kin.forward_kinematics(cfg)[target_joint_idx]
    assert T_world_target.shape == (7,)
    return jnp.concatenate(
        [
            T_world_target[4:7] - target.wxyz_xyz[4:7],
            # We're going to compute:
            #
            #    log(R_ee_target)
            #
            # (R_world_ee)^{-1} @ R_world_target
            (jaxlie.SO3(T_world_target[:4]).inverse() @ target.rotation()).log(),
        ]
    )


@jdc.jit
def solve_ik_analytically(
    kin: JaxKinTree,
    target_pose: jaxlie.SE3,
    target_joint_indices: tuple[int, ...],
    initial_pose: jnp.ndarray,
    JointVar: jdc.Static[type[jaxls.Var[jax.Array]]],
    ik_weight: jnp.ndarray,
    use_autodiff_jac: jdc.Static[bool] = False,
    idx_applied_to_target: jax.Array | None = None,
    *,
    joint_var_idx: int = 0,
    rest_weight: float = 0.001,
    limit_weight: float = 100.0,
    max_iterations: int = 50,
) -> jnp.ndarray:
    """
    Solve IK; this script is intended to benchmark autodiff vs. analytical
    jacobians. This function is similar to `solve_ik`, but it also takes in:

    - `use_autodiff_jac`: whether to use autodiff to compute the jacobian.
    If False, we use `jac_position_cost` to compute the jacobian analytically.

    - `idx_applied_to_target`: the set of actuated joints that were used to compute the jacobian.
    """
    # Add common factors.
    factors: list[jaxls.Factor] = [
        RobotFactors.limit_cost_factor(
            JointVar,
            joint_var_idx,
            kin,
            jnp.array([limit_weight] * kin.num_actuated_joints),
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
        return (
            position_and_orientation_cost(
                kin,
                vals[var],
                jaxlie.SE3(target_pose.wxyz_xyz.squeeze()),
                target_joint_idx,
            )
            * weights
        ).flatten()

    for i, target_joint_idx in enumerate(target_joint_indices):
        factors.append(
            jaxls.Factor(
                ik_cost,
                (
                    JointVar(joint_var_idx),
                    target_joint_idx,
                    ik_weight,
                ),
                "auto" if use_autodiff_jac else "custom",
                jac_custom_fn=(
                    None
                    if use_autodiff_jac or idx_applied_to_target is None
                    else lambda vals,
                    var,
                    target_joint_idx,
                    weights: jac_position_and_orientation_cost(
                        kin,
                        vals[var],
                        target_joint_idx,
                        idx_applied_to_target[i],
                        jaxlie.SE3(target_pose.wxyz_xyz.squeeze()),
                    )
                    * weights[:, None]
                ),
            ),
        )

    joint_vars: list[jaxls.Var] = [JointVar(joint_var_idx)]
    joint_var_values: list[jaxls.Var | jaxls._variables.VarWithValue] = [
        JointVar(joint_var_idx).with_value(initial_pose)
    ]
    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars,
        use_onp=False,
    )
    solution = graph.solve(
        linear_solver="dense_cholesky",
        initial_vals=jaxls.VarValues.make(joint_var_values),
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5,
            parameter_tolerance=1e-5,
            max_iterations=max_iterations,
        ),
        verbose=False,
    )

    joints = solution[JointVar(0)]
    return joints


def get_idx_applied_to_target(
    kin: JaxKinTree,
    target_joint_idx: int,
):
    # Get kinematic chain indices.
    def body_fun(i, curr_parent, idx):
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
    return idx_applied_to_target


@app.command()
def vis(
    robot_description: str,
) -> None:
    server = viser.ViserServer()

    urdf = load_urdf(robot_description)
    viser_urdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/urdf")
    kin = JaxKinTree.from_urdf(urdf)

    rest_pose = (kin.limits_lower + kin.limits_upper) / 2.0
    viser_urdf.update_cfg(np.array(rest_pose))

    # Add target transform controls
    target_tf = server.scene.add_transform_controls("target_transform", scale=0.2)
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
    autodiff_jac = server.gui.add_checkbox("Autodiff jacobian", initial_value=False)
    target_joint_idx = urdf.joint_names.index(target_joint_handle.value)
    idx_applied_to_target = (get_idx_applied_to_target(kin, target_joint_idx),)

    @target_joint_handle.on_update
    def _(_):
        nonlocal idx_applied_to_target
        target_joint_idx = urdf.joint_names.index(target_joint_handle.value)
        idx_applied_to_target = (get_idx_applied_to_target(kin, target_joint_idx),)

    Ts_world_joint = kin.forward_kinematics(rest_pose)
    target_joint_idx = urdf.joint_names.index(target_joint_handle.value)
    T_target = jaxlie.SE3(Ts_world_joint[target_joint_idx])

    target_tf.position = np.array(T_target.translation())
    target_tf.wxyz = np.array(T_target.rotation().wxyz)
    target_frame.position = np.array(T_target.translation())
    target_frame.wxyz = np.array(T_target.rotation().wxyz)

    # Add button to set target to current pose
    current_cfg = rest_pose

    has_jitted = False

    JointVar = RobotFactors.get_var_class(kin, rest_pose)

    timing = None
    while True:
        # current_cfg = jnp.array([slider.value for slider in slider_handles])
        target_pose = jaxlie.SE3(jnp.array([*target_tf.wxyz, *target_tf.position]))

        # Set IK parameters based on smooth mode
        # Solve IK with timing
        start_time = time.time()
        new_joints = solve_ik_analytically(
            kin,
            target_pose=target_pose,
            target_joint_indices=(target_joint_idx,),
            initial_pose=rest_pose,
            JointVar=JointVar,
            use_autodiff_jac=autodiff_jac.value,
            idx_applied_to_target=idx_applied_to_target,
            ik_weight=jnp.array([5.0] * 3 + [1.0] * 3),  # Position weights
            rest_weight=0.01,
            limit_weight=100.0,
        )

        # Ensure computation is complete before timing
        jax.block_until_ready(new_joints)
        if timing is None:
            timing = time.time() - start_time
        else:
            timing = 0.99 * timing + 0.01 * (time.time() - start_time)
        timing_handle.value = timing * 1000
        if not has_jitted:
            print(f"JIT compile + running took {timing:.1f} ms")
            has_jitted = True

        current_cfg = new_joints
        viser_urdf.update_cfg(np.array(current_cfg))

        Ts_world_joint = kin.forward_kinematics(current_cfg)
        target_joint_idx = urdf.joint_names.index(target_joint_handle.value)
        T_target = jaxlie.SE3(Ts_world_joint[target_joint_idx])
        target_frame.position = np.array(T_target.translation())
        target_frame.wxyz = np.array(T_target.rotation().wxyz)


robot_to_target_joint_names = {
    "panda": ["panda_hand_tcp_joint"],
}


@app.command()
def profile(
    robot_description: Literal["panda"] = "panda",
    n_trials: int = 100,
    batch_size: int = 100,
    use_autodiff_jac: bool = True,
):
    urdf = load_urdf(robot_description)
    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_lower + kin.limits_upper) / 2.0
    JointVar = RobotFactors.get_var_class(kin, rest_pose)

    # Get the target joint names.
    target_joint_name_list = robot_to_target_joint_names[robot_description]
    n_target_joints = len(target_joint_name_list)

    # Gather the target-to-act mapping for analytical jacobian.
    target_idx_list = []
    idx_applied_to_target_list = []
    for target_joint_name in target_joint_name_list:
        target_joint_idx = urdf.joint_names.index(target_joint_name)
        target_idx_list.append(target_joint_idx)
        idx_applied_to_target = get_idx_applied_to_target(kin, target_joint_idx)
        idx_applied_to_target_list.append(idx_applied_to_target)

    # sample target positions from random configurations.
    random_key = jax.random.PRNGKey(0)
    random_cfg = jax.random.uniform(
        random_key,
        (batch_size, kin.num_actuated_joints),
        minval=kin.limits_lower,
        maxval=kin.limits_upper,
    )
    random_T = jaxlie.SE3(kin.forward_kinematics(random_cfg)[..., target_idx_list, :])
    assert random_T.get_batch_axes() == (batch_size, n_target_joints)

    logger.info("Using {} jacobian", "autodiff" if use_autodiff_jac else "analytical")
    # Solve IK with analytical jacobian.
    vmap_fn = jax.jit(
        jax.vmap(
            lambda target_pose: solve_ik_analytically(
                kin,
                target_pose=target_pose,
                target_joint_indices=tuple(target_idx_list),
                initial_pose=rest_pose,
                JointVar=JointVar,
                use_autodiff_jac=use_autodiff_jac,
                idx_applied_to_target=jnp.array(idx_applied_to_target_list),
                ik_weight=jnp.array([5.0] * 3 + [1.0] * 3),  # Position weights
            ),
        )
    )
    start_time = time.time()
    joints = vmap_fn(random_T)
    jax.block_until_ready(joints)
    logger.info("Analytical jacobian took {:.1f} ms", time.time() - start_time)

    Ts_solved = jaxlie.SE3(kin.forward_kinematics(joints)[..., target_idx_list, :])
    errors = jnp.linalg.norm(Ts_solved.translation() - random_T.translation(), axis=-1)
    logger.info("Mean error: {}", errors.mean())

    # Time profiling.
    start_time = time.time()
    for _ in range(n_trials):
        joints = vmap_fn(random_T)
        jax.block_until_ready(joints)
    average_elapsed = (time.time() - start_time) / n_trials
    logger.info("Time per trial: {:.1f} ms", average_elapsed * 1000)
    return average_elapsed


@app.command()
def profile_batch(
    robot_description: Literal["panda"] = "panda",
    device: Literal["cpu", "gpu"] = "cpu",
    n_trials: int = 100,
):
    jax.config.update("jax_platform_name", device)
    logger.info("Using {} device", device)

    batch_size_list = [1, 10, 100, 1000]
    average_elapsed_no_autodiff = []
    average_elapsed_autodiff = []

    for batch_size in batch_size_list:
        average_elapsed_no_autodiff.append(
            profile(robot_description, n_trials, batch_size, use_autodiff_jac=False)
        )
        average_elapsed_autodiff.append(
            profile(robot_description, n_trials, batch_size, use_autodiff_jac=True)
        )

    import pandas as pd

    data = {
        "batch_size": batch_size_list,
        "average_elapsed_no_autodiff (ms)": [
            time * 1000 for time in average_elapsed_no_autodiff
        ],
        "average_elapsed_autodiff (ms)": [
            time * 1000 for time in average_elapsed_autodiff
        ],
    }

    df = pd.DataFrame(data)
    print(df.to_string(index=False))


if __name__ == "__main__":
    app.cli()
