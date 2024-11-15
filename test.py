import time
from typing import Literal, cast
import jax.numpy as jnp
from loguru import logger
import numpy as onp
import jax
from jaxmp.extras.grasp_antipodal import AntipodalGrasps
from jaxmp.extras import lock_joints
from pathlib import Path
import jaxlie
import jaxls
from jaxmp import JaxKinTree, RobotFactors
from jaxmp.coll import RobotColl, Sphere
from jaxmp.extras.rsrd_object import RSRDObject, RSRDVisualizer
from jaxmp.extras import load_urdf
import viser
from viser.extras import ViserUrdf
import jax_dataclasses as jdc


def ik_cost(
    vals: jaxls.VarValues,
    kin: JaxKinTree,
    var: jaxls.Var[jax.Array],
    var_offset: jaxls.Var[jaxlie.SE3],
    target_pose: jaxlie.SE3,
    target_joint_idx: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Pose cost."""
    joint_cfg: jax.Array = vals[var]
    pose_offset: jaxlie.SE3 = vals[var_offset]
    Ts_joint_world = kin.forward_kinematics(joint_cfg)
    residual = (
        (jaxlie.SE3(Ts_joint_world[target_joint_idx]) @ pose_offset).inverse() @ (target_pose)
    ).log()
    weights = jnp.broadcast_to(weights, residual.shape)
    return (residual * weights).flatten()


@jdc.jit
def solve_ik(
    kin: JaxKinTree,
    coll: RobotColl,
    world_coll: Sphere,
    target_joint_idx: jax.Array,
    target_pose: jaxlie.SE3,
    *,
    limit_weight: float = 100.0,
    rest_weight: float = 0.1,
) -> jax.Array:
    num_joints, timestep = target_pose.get_batch_axes()

    JointVar = RobotFactors.get_var_class(kin)

    def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
        delta = delta * jnp.zeros(6).at[3].set(1.0)
        return jaxls.SE3Var.retract_fn(transform, delta)

    class ConstrainedSE3Var(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.identity(batch_axes=(num_joints,)),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=retract_fn,
    ): ...

    factors = []
    for var_idx in range(timestep):
        factors.extend(
            [
                RobotFactors.limit_cost_factor(
                    JointVar,
                    var_idx,
                    kin,
                    jnp.full((kin.num_actuated_joints,), limit_weight),
                ),
                RobotFactors.rest_cost_factor(
                    JointVar, var_idx, jnp.full((kin.num_actuated_joints,), rest_weight)
                ),
            ]
        )
        ik_cost_factor = jaxls.Factor(
            ik_cost,
            (
                kin,
                JointVar(var_idx),
                ConstrainedSE3Var(0),
                jax.tree.map(lambda x: x[:, var_idx], target_pose),
                target_joint_idx,
                jnp.array([5.0] * 3 + [2.0] * 3),
            ),
        )
        if var_idx > 0:
            factors.append(
                jaxls.Factor(
                    lambda vals, var_0, var_1: (vals[var_0] - vals[var_1]).flatten() * 10.0,
                    (JointVar(var_idx), JointVar(var_idx - 1)),
                )
            )
        factors.append(ik_cost_factor)
        # factors.extend(
        #     RobotFactors.self_coll_factors(JointVar, var_idx, kin, coll, 0.01, 5.0)
        # )
        # factors.extend(
        #     RobotFactors.world_coll_factors(
        #         JointVar,
        #         var_idx,
        #         kin,
        #         coll,
        #         jax.tree.map(lambda x: x[var_idx, :], world_coll),
        #         0.001,
        #         jnp.full((len(coll.coll_link_names),), 1.0),
        #     )
        # )

    variables = [JointVar(var_idx) for var_idx in range(timestep)] + [ConstrainedSE3Var(0)]
    graph = jaxls.FactorGraph.make(
        factors,
        variables,
        use_onp=False,
    )
    solution = graph.solve(
        jaxls.VarValues.make(variables),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(max_iterations=40),
        verbose=False,
    )
    return jnp.stack([solution[JointVar(var_idx)] for var_idx in range(timestep)])


def get_joints(
    kin: JaxKinTree,
    coll: RobotColl,
    rsrd_obj: RSRDObject,
    T_obj_world: jaxlie.SE3,
    joint_indices: jax.Array,
    part_indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    assert rsrd_obj.grasps is not None
    num_grasps = rsrd_obj.grasps.centers.shape[-2]
    num_parts = len(part_indices)

    grasps = cast(
        AntipodalGrasps, jax.tree.map(lambda x: x[part_indices], rsrd_obj.grasps)
    )
    T_grasp_part = cast(
        jaxlie.SE3,
        jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=1),
            grasps.to_se3(),
            grasps.to_se3(flip_axis=True),
        ),
    )  # [num_parts, num_grasps, 7]
    assert T_grasp_part.get_batch_axes() == (num_parts, num_grasps * 2)

    T_part_obj = jaxlie.SE3(rsrd_obj.init_p2o[part_indices])
    T_part_part = jaxlie.SE3(rsrd_obj.part_deltas[:, part_indices])
    T_world_part = T_obj_world @ T_part_obj @ T_part_part  # [time, n_parts,]
    T_world_part = jaxlie.SE3(T_world_part.wxyz_xyz[..., None, :])  # [1, n_parts, 1]
    T_grasp_world = T_world_part @ T_grasp_part  # [time, n_parts, num_grasps * 2, 7]
    T_grasp_world = jaxlie.SE3(
        jnp.transpose(T_grasp_world.wxyz_xyz, (2, 1, 0, 3))
    )  # [num_grasps * 2, n_parts, time, 7]

    obj_point_list = []
    for part_idx in range(rsrd_obj.num_groups):
        part = rsrd_obj.get_part(part_idx)
        T_part_world = (
            T_obj_world
            @ jaxlie.SE3(rsrd_obj.init_p2o[part_idx])
            @ jaxlie.SE3(rsrd_obj.part_deltas[:, part_idx])
        )
        T_part_world = jax.tree.map(lambda x: x[:, None, ...], T_part_world)
        part_points = T_part_world @ (part.means)
        obj_point_list.append(part_points)
    obj_points = jnp.concatenate(obj_point_list, axis=1)
    obj_points = obj_points[:, ::100, :]
    world_coll = Sphere.from_center_and_radius(
        obj_points, jnp.full((*obj_points.shape[:-1], 1), 0.0)
    )

    joints = jax.vmap(solve_ik, in_axes=(None, None, None, None, 0))(
        kin,
        coll,
        world_coll,
        joint_indices,
        T_grasp_world,
    )  # [n_grasps, timestep, joints]

    T_target_world = kin.forward_kinematics(joints)[
        ..., joint_indices, :
    ]  # [n_grasps, timesteps, num_target_joints, 7]
    T_target_world = jnp.transpose(T_target_world, (0, 2, 1, 3))
    dist = jnp.linalg.norm(
        (jaxlie.SE3(T_target_world).inverse() @ T_grasp_world).log()[..., :3],
        axis=-1,
    )
    success = (dist < 0.01).all(axis=(1, 2))
    return joints, success


def main(
    data_path: Path = Path("examples/assets/rsrd/scissors_rsrd.txt"),
    mode: Literal["single", "bimanual"] = "bimanual",
):
    prng_key = jax.random.PRNGKey(0)
    rsrd_obj = RSRDObject.from_data(data_path, prng_key)

    with jdc.copy_and_mutate(rsrd_obj, validate=False) as rsrd_obj:
        rsrd_obj.part_deltas = rsrd_obj.part_deltas[::5]
        rsrd_obj.timesteps = rsrd_obj.part_deltas.shape[0]

    assert rsrd_obj.grasps is not None

    server = viser.ViserServer()
    tf_handle = server.scene.add_transform_controls("/object", scale=0.3)
    rsrd_vis = RSRDVisualizer(server, rsrd_obj, base_frame_name="/object")

    # for i in range(rsrd_obj.num_groups):
    #     part_frame = rsrd_vis.get_part_frame_name(i)
    #     part_grasps = jax.tree.map(lambda x: x[i], rsrd_obj.grasps)
    #     server.scene.add_mesh_trimesh(part_frame + "/grasps", part_grasps.to_trimesh())

    urdf = load_urdf(
        robot_urdf_path=Path("../../please2/toad/data/yumi_description/urdf/yumi.urdf")
    )
    urdf = lock_joints(
        urdf,
        [
            "gripper_r_joint",
            "gripper_l_joint",
            "gripper_r_joint_m",
            "gripper_l_joint_m",
        ],
        [0.025] * 4,
    )
    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    self_coll_ignore = [
        (k1, k2) for k1 in urdf.link_map.keys() for k2 in urdf.link_map.keys()
    ]
    keywords = ["gripper"]
    self_coll_ignore = list(
        filter(
            lambda x: not any(k in x[0] or k in x[1] for k in keywords),
            self_coll_ignore,
        )
    )

    def coll_handler(meshes):
        spheres = [Sphere.from_min_sphere(mesh) for mesh in meshes]
        return jax.tree.map(lambda *x: jnp.stack(x), *spheres)

    world_coll_ignore = list(urdf.link_map.keys())
    world_coll_ignore = list(
        filter(lambda x: not any(k in x for k in keywords), world_coll_ignore)
    )
    coll = RobotColl.from_urdf(
        urdf,
        self_coll_ignore=self_coll_ignore,
        coll_handler=coll_handler,
        world_coll_ignore=world_coll_ignore,
    )

    left_joint = kin.joint_names.index("left_dummy_joint")
    right_joint = kin.joint_names.index("right_dummy_joint")

    urdf_vis = ViserUrdf(server, urdf)
    urdf_vis.update_cfg(onp.array(rest_pose))

    succ_traj = jnp.repeat(rest_pose[None, :], rsrd_obj.timesteps, axis=0)

    def get_traj():
        nonlocal succ_traj
        assert rsrd_obj.single_hand_assignments is not None
        assert rsrd_obj.bimanual_assignments is not None

        T_obj_world = jaxlie.SE3(jnp.array([*tf_handle.wxyz, *tf_handle.position]))

        _succ_traj = None
        for idx in range(len(rsrd_obj.bimanual_assignments)):
            for hand_idx in range(2):
                if hand_idx == 0:
                    joint_indices = jnp.array([left_joint, right_joint])
                else:
                    joint_indices = jnp.array([right_joint, left_joint])

                part_indices = jnp.array(rsrd_obj.bimanual_assignments[idx])
                logger.info(f"Trying bimanual assignment, with parts {part_indices}.")

                start = time.time()
                joints, success = get_joints(
                    kin, coll, rsrd_obj, T_obj_world, joint_indices, part_indices
                )
                logger.info(f"Took {time.time() - start} seconds.")

                joints = joints[success]
                if len(joints) > 0:
                    _succ_traj = joints[0]
                    break
            if _succ_traj is not None:
                break
        else:
            raise ValueError("No valid trajectory found.")
        succ_traj = _succ_traj

    traj_gen_button = server.gui.add_button("Generate Trajectory")

    @traj_gen_button.on_click
    def _(_):
        traj_gen_button.disabled = True
        # server.scene.add_point_cloud("points", onp.array(obj_points[0]), colors=onp.zeros_like(obj_points[0]), point_size=0.001)
        # breakpoint()
        get_traj()
        traj_gen_button.disabled = False

    timestep_handler = server.gui.add_slider(
        "Timestep", 0, rsrd_obj.timesteps - 1, 1, 0
    )
    play_handler = server.gui.add_checkbox("Play", True)
    while True:
        timestep = timestep_handler.value
        if play_handler.value:
            timestep_handler.value = (timestep + 1) % rsrd_obj.timesteps
        urdf_vis.update_cfg(onp.array(succ_traj)[timestep])
        rsrd_vis.update_cfg(timestep)
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
