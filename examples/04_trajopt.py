"""04_trajopt.py
Given some SE3 trajectory in joint frame, optimize the robot joint trajectory for path smoothness.
Also supports "GOMP"-like features, where you can free certain axes of the target frame.
"""

import time
from pathlib import Path
import jax
from robot_descriptions.loaders.yourdfpy import load_robot_description
import viser
import viser.extras

import jax.numpy as jnp
import jaxlie
import numpy as onp

import jax_dataclasses as jdc
import jaxls

from jaxmp.kinematics import JaxKinTree
from jaxmp.robot_factors import RobotFactors


@jdc.jit
def solve_traj_gomp(
    kin: JaxKinTree,
    target_pose: jaxlie.SE3,
    target_joint_indices: jax.Array,
    pos_weight: float,
    rot_weight: float,
    rest_weight: float,
    limit_weight: float,
    smoothness_weight: float,
    rest_pose: jnp.ndarray,
    freeze_target_xyz_xyz: jnp.ndarray,
):
    # Create factor graph.
    JointVar = RobotFactors.get_var_class(kin, default_val=rest_pose)

    def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
        delta = delta * (1 - freeze_target_xyz_xyz)
        return jaxls.SE3Var.retract_fn(transform, delta)

    class ConstrainedSE3Var(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.identity(),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=retract_fn,
    ): ...

    def ik_cost(
        vals: jaxls.VarValues,
        kin: JaxKinTree,
        joint_var: jaxls.Var[jax.Array],
        target_pose: jaxlie.SE3,
        target_pose_offset_var: ConstrainedSE3Var,
        target_joint_idx: jdc.Static[int],
        weights: jax.Array,
    ) -> jax.Array:
        """Pose cost."""
        joint_cfg: jax.Array = vals[joint_var]
        target_pose_offset = vals[target_pose_offset_var]
        Ts_joint_world = kin.forward_kinematics(joint_cfg)
        residual = (
            (jaxlie.SE3(Ts_joint_world[target_joint_idx])).inverse()
            @ (target_pose @ target_pose_offset)
        ).log()
        weights = jnp.broadcast_to(weights, residual.shape)
        assert residual.shape == weights.shape
        return (residual * weights).flatten()

    _, timesteps, pose_dim = target_pose.wxyz_xyz.shape
    factors = []

    for tstep in range(timesteps):
        for idx, target_joint_idx in enumerate(target_joint_indices):
            factors.extend(
                [
                    jaxls.Factor(
                        ik_cost,
                        (
                            kin,
                            JointVar(tstep),
                            jaxlie.SE3(target_pose.wxyz_xyz[idx, tstep]),
                            ConstrainedSE3Var(idx),
                            target_joint_idx,
                            jnp.array([pos_weight] * 3 + [rot_weight] * 3),
                        ),
                    ),
                ]
            )
        factors.extend(
            [
                RobotFactors.limit_cost_factor(
                    JointVar,
                    tstep,
                    kin,
                    jnp.array([limit_weight] * kin.num_actuated_joints),
                ),
                RobotFactors.rest_cost_factor(
                    JointVar,
                    tstep,
                    jnp.array([rest_weight] * kin.num_actuated_joints),
                ),
            ]
        )
        if tstep > 0:
            factors.append(
                RobotFactors.smoothness_cost_factor(
                    JointVar,
                    tstep,
                    tstep - 1,
                    jnp.array([smoothness_weight] * kin.num_actuated_joints),
                )
            )

    traj_vars = [
        JointVar(jnp.arange(timesteps)),
        ConstrainedSE3Var(jnp.arange(len(rest_pose))),
    ]
    graph = jaxls.FactorGraph.make(
        factors,
        traj_vars,
        use_onp=False,
    )
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(traj_vars),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.2),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5, parameter_tolerance=1e-5
        ),
    )
    return jnp.stack([solution[JointVar(tstep)] for tstep in range(timesteps)])


def main(
    pos_weight: float = 10.0,
    rot_weight: float = 2.0,
    limit_weight: float = 100.0,
    rest_weight: float = 0.01,
    smoothness_weight: float = 10.0,
):
    server = viser.ViserServer()
    urdf = load_robot_description("yumi_description")
    urdf_orig = viser.extras.ViserUrdf(server, urdf, root_node_name="/urdf")
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    kin = JaxKinTree.from_urdf(urdf)
    trajectory = onp.load(
        Path(__file__).parent / "assets/yumi_trajectory.npy", allow_pickle=True
    ).item()  # {'joint_name': [time, wxyz_xyz]}
    timesteps = list(trajectory.values())[0].shape[0]
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    # Solve trajectory optimization.
    target_joint_indices = jnp.array(
        [kin.joint_names.index(k) for k in trajectory.keys()]
    )
    # Add trajectory visualization.
    target_pose = jaxlie.SE3(jnp.stack([v for v in trajectory.values()]))
    traj_handle = server.scene.add_transform_controls("traj_handle", scale=0.2)
    traj_center = target_pose.translation().reshape(-1, 3).mean(axis=0)
    traj_handle.position = onp.array(traj_center)
    for joint_name, joint_pose_traj in trajectory.items():
        trajectory[joint_name][..., 4:] -= traj_center
        server.scene.add_batched_axes(
            f"traj_handle/{joint_name}",
            batched_positions=joint_pose_traj[:, 4:],
            batched_wxyzs=joint_pose_traj[:, :4],
            axes_length=0.04,
            axes_radius=0.004,
        )

    freeze_target_xyz_xyz = jnp.ones(6)
    traj = solve_traj_gomp(
        kin,
        target_pose,
        target_joint_indices,
        pos_weight=pos_weight,
        rot_weight=rot_weight,
        rest_weight=rest_weight,
        limit_weight=limit_weight,
        smoothness_weight=smoothness_weight,
        rest_pose=rest_pose,
        freeze_target_xyz_xyz=freeze_target_xyz_xyz,
    )

    with server.gui.add_folder("Target frame"):
        freeze_target_x = server.gui.add_checkbox("Freeze x", initial_value=True)
        freeze_target_y = server.gui.add_checkbox("Freeze y", initial_value=True)
        freeze_target_z = server.gui.add_checkbox("Freeze z", initial_value=True)
        freeze_target_rx = server.gui.add_checkbox("Freeze rx", initial_value=True)
        freeze_target_ry = server.gui.add_checkbox("Freeze ry", initial_value=True)
        freeze_target_rz = server.gui.add_checkbox("Freeze rz", initial_value=True)

    update_traj_handle = server.gui.add_button("Regenerate traj")

    @update_traj_handle.on_click
    def _(_):
        nonlocal traj
        update_traj_handle.disabled = True

        freeze_target_xyz_xyz = jnp.array(
            [
                freeze_target_x.value,
                freeze_target_y.value,
                freeze_target_z.value,
                freeze_target_rx.value,
                freeze_target_ry.value,
                freeze_target_rz.value,
            ]
        ).astype(jnp.float32)

        target_pose = jaxlie.SE3(jnp.stack([v for v in trajectory.values()]))
        traj_center = target_pose.translation().reshape(-1, 3).mean(axis=0)
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(jnp.array(traj_handle.wxyz)),
            translation=jnp.array(traj_handle.position),
        ) @ jaxlie.SE3(
            target_pose.wxyz_xyz.at[..., 4:].set(
                target_pose.wxyz_xyz[..., 4:] - traj_center
            )
        )
        traj = solve_traj_gomp(
            kin,
            target_pose,
            target_joint_indices,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            rest_weight=rest_weight,
            limit_weight=limit_weight,
            smoothness_weight=smoothness_weight,
            rest_pose=rest_pose,
            freeze_target_xyz_xyz=freeze_target_xyz_xyz,
        )
        update_traj_handle.disabled = False

    # Visualize!
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )

    @slider.on_update
    def _(_) -> None:
        urdf_orig.update_cfg(onp.array(traj[slider.value]))

        Ts_world_joint = onp.array(kin.forward_kinematics(traj[slider.value]))
        for idx, joint_name in zip(target_joint_indices, trajectory.keys()):
            server.scene.add_frame(
                f"/joints/{joint_name}",
                wxyz=Ts_world_joint[idx, :4],
                position=Ts_world_joint[idx, 4:7],
                axes_length=0.1,
                axes_radius=0.01,
            )

    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    main()
