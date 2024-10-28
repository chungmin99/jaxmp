from pathlib import Path
import time
import viser
from viser.extras import ViserUrdf

import numpy as onp
import jax.numpy as jnp
import jaxlie

from jaxmp import JaxKinTree, RobotFactors
from jaxmp.coll import RobotColl, Sphere, Capsule
from jaxmp.extras import load_urdf, lock_joints, AntipodalGrasps

from typing import Literal, Optional

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import jax_dataclasses as jdc

from jaxmp.robot_factors import RobotFactors
from jaxmp.kinematics import JaxKinTree
from jaxmp.coll import RobotColl, CollGeom

def ik_cost(
    vals: jaxls.VarValues,
    kin: JaxKinTree,
    var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_joint_idx: jax.Array,
    weights: jax.Array,
    base_tf_var: jaxls.Var[jaxlie.SE3] | jaxlie.SE3 | None = None,
) -> jax.Array:
    """Pose cost."""
    joint_cfg: jax.Array = vals[var]
    if isinstance(base_tf_var, jaxls.Var):
        base_tf = vals[base_tf_var]
    elif isinstance(base_tf_var, jaxlie.SE3):
        base_tf = base_tf_var
    else:
        base_tf = jaxlie.SE3.identity()

    Ts_joint_world = kin.forward_kinematics(joint_cfg)
    residual = (
        (base_tf @ jaxlie.SE3(Ts_joint_world[target_joint_idx])).inverse()
        @ (target_pose)
    ).log()
    weights = jnp.broadcast_to(weights, residual.shape)
    assert residual.shape == weights.shape
    residual = (residual * weights)
    residual = residual[jnp.argmin(jnp.abs(residual).sum(axis=-1))]
    return residual.flatten()


@jdc.jit
def solve_ik(
    kin: JaxKinTree,
    target_pose: jaxlie.SE3,
    target_joint_indices: jax.Array,
    rest_pose: jnp.ndarray,
    *,
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    manipulability_weight: float = 0.0,
    include_manipulability: jdc.Static[bool] = False,
    joint_vel_weight: float = 0.0,
    self_coll_weight: float = 2.0,
    world_coll_weight: float = 10.0,
    robot_coll: Optional[RobotColl] = None,
    include_self_coll: jdc.Static[bool] = False,
    world_coll_list: list[CollGeom] = [],
    solver_type: jdc.Static[Literal[
        "cholmod", "conjugate_gradient", "dense_cholesky"
    ]] = "conjugate_gradient",
    freeze_target_xyz_xyz: Optional[jnp.ndarray] = None,
    freeze_base_xyz_xyz: Optional[jnp.ndarray] = None,
    dt: float = 0.1,
) -> tuple[jaxlie.SE3, jnp.ndarray]:
    """
    Solve IK for the robot.
    Args:
        target_pose: Desired pose of the target joint, SE3 has batch axes (n_target,).
        target_joint_indices: Indices of the target joints, length n_target.
        freeze_target_xyz_xyz: 6D vector indicating which axes to freeze in the target frame.
        freeze_base_xyz_xyz: 6D vector indicating which axes to freeze in the base frame.
    Returns:
        Base pose (jaxlie.SE3)
        Joint angles (jnp.ndarray)
    """
    if freeze_target_xyz_xyz is None:
        freeze_target_xyz_xyz = jnp.ones(6)
    if freeze_base_xyz_xyz is None:
        freeze_base_xyz_xyz = jnp.ones(6)

    JointVar = RobotFactors.get_var_class(kin, default_val=rest_pose)
    ConstrainedSE3Var = RobotFactors.get_constrained_se3(freeze_base_xyz_xyz)

    joint_vars = [JointVar(0), ConstrainedSE3Var(0)]

    factors: list[jaxls.Factor] = [
        jaxls.Factor(
            RobotFactors.limit_cost,
            (
                kin,
                JointVar(0),
                jnp.array([limit_weight] * kin.num_actuated_joints),
            ),
        ),
        jaxls.Factor(
            RobotFactors.joint_limit_vel_cost,
            (
                kin,
                JointVar(0),
                rest_pose,
                dt,
                jnp.array([joint_vel_weight] * kin.num_actuated_joints),
            ),
        ),
        jaxls.Factor(
            RobotFactors.rest_cost,
            (
                JointVar(0),
                jnp.array([rest_weight] * kin.num_actuated_joints),
            ),
        ),
    ]

    ik_weights = jnp.array([pos_weight] * 3 + [rot_weight] * 3)
    ik_weights = ik_weights * freeze_target_xyz_xyz
    
    factors.append(
        jaxls.Factor(
            ik_cost,
            (
                kin,
                joint_vars[0],
                target_pose,
                target_joint_indices,
                ik_weights,
                ConstrainedSE3Var(0),
            ),
        ),
    )
    
    if include_manipulability:
        for idx, target_joint_idx in enumerate(target_joint_indices):
            factors.append(
                jaxls.Factor(
                    RobotFactors.manipulability_cost,
                    (
                        kin,
                        joint_vars[0],
                        target_joint_idx,
                        jnp.array([manipulability_weight] * kin.num_actuated_joints),
                    ),
                )
            )

    if robot_coll is not None:
        factors.append(
            jaxls.Factor(
                RobotFactors.self_coll_cost,
                (
                    kin,
                    robot_coll,
                    JointVar(0),
                    0.01,
                    jnp.full(robot_coll.coll.get_batch_axes(), self_coll_weight),
                    ConstrainedSE3Var(0),
                ),
            ),
        )
        for world_coll in world_coll_list:
            factors.append(
                jaxls.Factor(
                    RobotFactors.world_coll_cost,
                    (
                        kin,
                        robot_coll,
                        JointVar(0),
                        world_coll,
                        jnp.array([0.005]),
                        jnp.full(robot_coll.coll.get_batch_axes(), world_coll_weight),
                        ConstrainedSE3Var(0),
                    ),
                ),
            )

    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars,
        use_onp=False,
    )
    solution = graph.solve(
        linear_solver=solver_type,
        initial_vals=jaxls.VarValues.make(joint_vars),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5,
            parameter_tolerance=1e-5,
            max_iterations=40,
        ),
        verbose=False,
    )

    # Update visualization.
    base_pose = solution[ConstrainedSE3Var(0)]
    joints = solution[JointVar(0)]
    return base_pose, joints


if __name__ == "__main__":
    urdf = load_urdf("panda")
    urdf = lock_joints(
        urdf,
        ["panda_finger_joint1", "panda_finger_joint2"],
        [0.04] * 2,
    )

    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2
    coll = RobotColl.from_urdf(urdf)
    # obj = Sphere.from_center_and_radius(jnp.zeros(3), jnp.array([0.015]))
    obj = Capsule.from_radius_and_height(
        jnp.array([0.02]), jnp.array([0.1]), jaxlie.SE3.identity()
    )
    grasps = AntipodalGrasps.from_sample_mesh(obj.to_trimesh(), 100)

    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf)
    obj_handle = server.scene.add_transform_controls("obj", scale=0.2)
    server.scene.add_mesh_trimesh("obj/mesh", obj.to_trimesh())

    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0],
    )
    target_frame_handle = server.scene.add_frame("target", axes_length=0.1)

    joints = rest_pose
    while True:
        target_joint_indices = jnp.array(
            [
                kin.joint_names.index(target_name_handle.value)
            ]
        )
        target_poses = jaxlie.SE3(
            jnp.array([*obj_handle.wxyz, *obj_handle.position])
        )
        T_grasps = grasps.to_se3()
        T_grasps = jaxlie.SE3(jnp.concatenate([
            T_grasps.wxyz_xyz,
            (T_grasps @ jaxlie.SE3.from_rotation(jaxlie.SO3.from_z_radians(jnp.pi))).wxyz_xyz
        ]))
        target_poses = target_poses @ T_grasps

        curr_sphere_obs = obj.transform(
            jaxlie.SE3(
                jnp.array([*obj_handle.wxyz, *obj_handle.position])
            )
        )

        _, joints = solve_ik(
            kin,
            target_poses,
            target_joint_indices,
            joints,
            robot_coll=coll,
            world_coll_list=[curr_sphere_obs],
            self_coll_weight=0.0,
            world_coll_weight=1.0,
            include_self_coll=False,
            joint_vel_weight=1.0,
            freeze_target_xyz_xyz=jnp.ones(6).at[4].set(0),
        )

        urdf_vis.update_cfg(onp.array(joints))
