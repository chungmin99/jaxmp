"""05_trajopt.py
Given some SE3 trajectory in joint frame, optimize the robot joint trajectory for path smoothness.
"""

import time
from pathlib import Path
import jax
from loguru import logger
import viser
import viser.extras

import jax.numpy as jnp
import jaxlie
import numpy as onp

import jax_dataclasses as jdc

import pyroki as pk
import pyroki.viewer as viewer


@jdc.jit
def solve_trajopt(
    robot: pk.Robot,
    target_pose: jaxlie.SE3,
    target_link_indices: jax.Array,
    pos_weight: float,
    rot_weight: float,
    rest_weight: float,
    limit_weight: float,
    smoothness_weight: float,
    rest_pose: jnp.ndarray,
):
    # Define variables for each timestep
    _, timesteps, _ = target_pose.wxyz_xyz.shape
    traj_var = robot.JointVar(jnp.arange(timesteps))

    ik_weights = jnp.array([pos_weight] * 3 + [rot_weight] * 3)
    limit_weights = jnp.array([limit_weight] * robot.joint.count)
    rest_weights = jnp.array([rest_weight] * robot.joint.actuated_count)
    smoothness_weights = jnp.array([smoothness_weight] * robot.joint.actuated_count)

    target_poses_T_N_7_params = jnp.swapaxes(target_pose.wxyz_xyz, 0, 1)

    factors = [
        pk.PoseCost(
            (
                traj_var,
                jaxlie.SE3(target_poses_T_N_7_params),
            ),
            weights=ik_weights,
            target_link_indices=target_link_indices,
            robot=robot,
        ),
        pk.LimitCost(
            (traj_var,),
            weights=limit_weights,
            robot=robot,
        ),
        pk.RestCost(
            (traj_var,),
            rest_pose=rest_pose,
            weights=rest_weights,
        ),
        pk.SmoothnessCost(
            (
                robot.JointVar(jnp.arange(1, timesteps)),
                robot.JointVar(jnp.arange(0, timesteps - 1)),
            ),
            weights=smoothness_weights,
        ),
    ]

    solution, _ = pk.solve(
        vars=[traj_var],
        factors=factors,
        init_vars=[],
        verbose=False,
        max_iterations=50,
    )

    return solution[traj_var]


def main(
    pos_weight: float = 10.0,
    rot_weight: float = 2.0,
    limit_weight: float = 100.0,
    rest_weight: float = 0.01,
    smoothness_weight: float = 10.0,
):
    server = viser.ViserServer()

    urdf, robot = pk.load_robot(robot_description="yumi")
    robot_viz = viewer.BatchedURDF(server, urdf, root_node_name="/urdf")
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Load a dummy trajectory.
    trajectory = onp.load(
        Path(__file__).parent / "assets/yumi_trajectory.npy", allow_pickle=True
    ).item()  # {'joint_name': [time, wxyz_xyz]}
    timesteps = list(trajectory.values())[0].shape[0]
    rest_pose = (robot.joint.upper_limits_act + robot.joint.lower_limits_act) / 2

    # Solve trajectory optimization.
    target_link_indices = jnp.array(
        [robot.link.names.index(k) for k in trajectory.keys()]
    )
    target_pose = jaxlie.SE3(jnp.stack([v for v in trajectory.values()]))
    traj_handle = server.scene.add_transform_controls("traj_handle", scale=0.2)
    traj_center = target_pose.translation().reshape(-1, 3).mean(axis=0)
    traj_handle.position = onp.array(traj_center)
    for joint_name, joint_pose_traj in trajectory.items():
        joint_pose_traj_viz = joint_pose_traj.copy()
        joint_pose_traj_viz[..., 4:] -= traj_center
        server.scene.add_batched_axes(
            f"traj_handle/{joint_name}",
            batched_positions=joint_pose_traj_viz[:, 4:],
            batched_wxyzs=joint_pose_traj_viz[:, :4],
            axes_length=0.04,
            axes_radius=0.004,
        )

    update_traj_handle = server.gui.add_button("Regenerate traj")

    def generate_traj():
        nonlocal traj
        update_traj_handle.disabled = True

        target_pose_current = jaxlie.SE3(jnp.stack([v for v in trajectory.values()]))
        traj_center = target_pose_current.translation().reshape(-1, 3).mean(axis=0)

        T_world_handle = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(jnp.array(traj_handle.wxyz)),
            translation=jnp.array(traj_handle.position),
        )
        target_pose_centered = jaxlie.SE3(
            target_pose_current.wxyz_xyz.at[..., 4:].add(-traj_center)
        )
        target_pose_final = T_world_handle @ target_pose_centered

        start = time.time()
        traj = solve_trajopt(
            robot,
            target_pose_final,
            target_link_indices,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            rest_weight=rest_weight,
            limit_weight=limit_weight,
            smoothness_weight=smoothness_weight,
            rest_pose=rest_pose,
        )
        end = time.time()
        logger.info(f"Trajectory optimization took {end - start:.2f}s")
        update_traj_handle.disabled = False

    update_traj_handle.on_click(lambda _: generate_traj())

    traj = None
    generate_traj()

    # Visualize!
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )

    @slider.on_update
    def _(_) -> None:
        assert traj is not None
        robot_viz.update_cfg(traj[slider.value])

        Ts_world_link = onp.array(robot.forward_kinematics(traj[slider.value]))
        for idx, joint_name in zip(target_link_indices, trajectory.keys()):
            server.scene.add_frame(
                f"/joints/{joint_name}",
                wxyz=Ts_world_link[idx, :4],
                position=Ts_world_link[idx, 4:7],
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
