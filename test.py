import time
from typing import cast
import jax.numpy as jnp
import numpy as onp
import jax
from jaxmp.extras.grasp_antipodal import AntipodalGrasps
import trimesh
from pathlib import Path
import jaxlie
import jaxls
from jaxmp import JaxKinTree, RobotFactors
from jaxmp.extras.rsrd_object import RSRDObject, RSRDVisualizer
from jaxmp.extras import load_urdf
import viser
from viser.extras import ViserUrdf
import jax_dataclasses as jdc

def ik_cost(
    vals: jaxls.VarValues,
    kin: JaxKinTree,
    var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_joint_idx: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Pose cost."""
    joint_cfg: jax.Array = vals[var]
    Ts_joint_world = kin.forward_kinematics(joint_cfg)
    residual = (
        (jaxlie.SE3(Ts_joint_world[target_joint_idx])).inverse()
        @ (target_pose)
    ).log()
    weights = jnp.broadcast_to(weights, residual.shape)
    assert residual.shape == weights.shape
    # residual = (0.1 * residual * weights) / (joint_cfg.shape[0])
    residual = residual[jnp.argmin(jnp.abs(residual).sum(axis=-1))]
    return residual.flatten()


@jdc.jit
def solve_ik(
    kin: JaxKinTree,
    target_joint_idx: jax.Array,
    target_pose: jaxlie.SE3,
    *,
    limit_weight: float = 100.0,
    rest_weight: float = 0.01,
) -> jax.Array:
    JointVar = RobotFactors.get_var_class(kin)
    var_idx = jnp.array(0)
    factors = []
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
            target_pose,
            target_joint_idx,
            jnp.array([5.0]*3 + [1.0]*3).at[4].set(0.0),
        )
    )
    factors.append(ik_cost_factor)

    variables = [JointVar(var_idx)]
    graph = jaxls.FactorGraph.make(
        factors,
        variables,
        use_onp=False,
    )
    solution = graph.solve(
        jaxls.VarValues.make(variables),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        verbose=False,
    )
    return solution[JointVar(var_idx)]


def main(data_path: Path = Path("examples/assets/rsrd/scissors_rsrd.txt")):
    prng_key = jax.random.PRNGKey(0)
    rsrd_obj = RSRDObject.from_data(data_path, prng_key)

    server = viser.ViserServer()
    tf_handle = server.scene.add_transform_controls("/object", scale=0.3)
    rsrd_vis = RSRDVisualizer(server, rsrd_obj, base_frame_name="/object")

    server.scene.add_mesh_trimesh("grasps", rsrd_obj.grasps.to_trimesh())

    breakpoint()

    # urdf = load_urdf(robot_urdf_path=Path("../../please2/toad/data/yumi_description/urdf/yumi.urdf"))
    # kin = JaxKinTree.from_urdf(urdf)
    # left_joint = kin.joint_names.index("left_dummy_joint")
    # right_joint = kin.joint_names.index("right_dummy_joint")
    # urdf_vis = ViserUrdf(server, urdf)

    # assert rsrd_obj.single_hand_assignments is not None
    # part_idx = rsrd_obj.single_hand_assignments[0]
    # grasps = cast(AntipodalGrasps, jax.tree.map(lambda x: x[part_idx], rsrd_obj.grasps))
    # T_grasp_part = cast(
    #     jaxlie.SE3,
    #     jax.tree.map(
    #         lambda *x: jnp.concatenate(x),
    #         grasps.to_se3(),
    #         grasps.to_se3(flip_axis=True),
    #     ),
    # )

    # while True:
    #     T_obj_world = jaxlie.SE3(jnp.array([*tf_handle.wxyz, *tf_handle.position]))
    #     T_part_obj = jaxlie.SE3(rsrd_obj.part_deltas[0, part_idx])
    #     T_grasp_world = T_obj_world @ T_part_obj @ T_grasp_part
    
    #     joints = solve_ik(
    #         kin,
    #         jnp.array([left_joint]),
    #         T_grasp_world,
    #     )
    #     urdf_vis.update_cfg(onp.array(joints))


    # solve_ik(
    #     kin, jnp.array([left_joint, right_joint]), 
    # )

    # timestep = 0
    # while True:
    #     timestep = (timestep + 1) % rsrd_obj.timesteps
    #     rsrd_vis.update_cfg(timestep)
    #     time.sleep(0.01)

if __name__ == "__main__":
    main()
