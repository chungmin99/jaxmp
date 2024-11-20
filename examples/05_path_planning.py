""" 05_path_planning.py
Uses geometric path planning to find a path from start to goal.
"""

from jaxmp.coll._collide_types import Capsule
import viser
from viser.extras import ViserUrdf

import jax
import jax.numpy as jnp
import numpy as onp
import jaxlie
import jax_dataclasses as jdc

import mctx

from jaxmp import JaxKinTree, RobotFactors
from jaxmp.coll import RobotColl, Sphere, collide
from jaxmp.extras import load_urdf, solve_ik

from jaxmp.path._mcts_types import SearchParams, SearchState

@jdc.jit
def step(params, start_state, prng_key):
    root = mctx.RootFnOutput(
        prior_logits=jnp.zeros([1, params.n_actions]),
        value=start_state.get_dist_heuristic(params.target_state),
        embedding=start_state,
    )
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=prng_key,
        root=root,
        recurrent_fn=params.get_recurrent_fn(),
        num_simulations=10,
        max_depth=params.max_steps,
    )
    return policy_output

@jdc.jit
def foo(params, start_state, prng_key, coll, kin, curr_sphere_obs):
    def dist_fn(state, target):
        dist_value = -jnp.linalg.norm(state.value - target.value, axis=-1)
        coll_dist = (
            coll.world_coll_dist(kin, state.value, curr_sphere_obs)
            .min()
            .clip(max=0.0)
        )
        return dist_value + coll_dist * 10

    root = mctx.RootFnOutput(
        prior_logits=jnp.zeros([1, params.n_actions]),
        value=start_state.get_dist_heuristic(params.target_state),
        embedding=start_state,
    )

    recurrent_fn = params.get_recurrent_fn(dist_fn)
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=prng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=100,
        max_depth=params.max_steps,
    )
    return policy_output

def main():
    urdf = load_urdf("panda")
    kin = JaxKinTree.from_urdf(urdf)
    coll = RobotColl.from_urdf(urdf)
    rest_joints = (kin.limits_upper + kin.limits_lower) / 2
    JointVar = RobotFactors.get_var_class(kin)
    ik_weight = jnp.array([5.0]*3 + [1.0]*3)
    
    server = viser.ViserServer()

    curr_joints = rest_joints.copy()

    # Robot being moved with IK.
    urdf_vis = ViserUrdf(server, urdf)
    urdf_vis.update_cfg(onp.array(curr_joints))

    # Robot following with path planning.
    urdf_vis_path = ViserUrdf(server, urdf, root_node_name="/path")
    urdf_vis_path.update_cfg(onp.array(curr_joints))

    # GUI elements to control IK elements.
    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0],
    )
    target_tf_handle = server.scene.add_transform_controls(
        "target_transform", scale=0.2
    )

    sphere_obs = Capsule.from_radius_and_height(jnp.array([0.05]), jnp.array([2.0]), jaxlie.SE3.identity())
    # sphere_obs = Sphere.from_center_and_radius(jnp.zeros(3), jnp.array([0.05]))
    sphere_obs_handle = server.scene.add_transform_controls(
        "sphere_obs", scale=0.2, position=(0.2, 0.0, 0.2)
    )
    server.scene.add_mesh_trimesh("sphere_obs/mesh", sphere_obs.to_trimesh())

    target_joints = rest_joints
    while True:
        n_actions = 30

        # Get the target joints.
        target_joint_indices = jnp.array([kin.joint_names.index(target_name_handle.value)])
        target_poses = jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))
        _, target_joints = solve_ik(
            kin, target_poses, target_joint_indices, target_joints, JointVar, ik_weight
        )
        urdf_vis.update_cfg(onp.array(target_joints))

        start_state = SearchState.from_value(curr_joints[None])
        target_state = SearchState.from_value(target_joints[None])

        prng_key = jax.random.PRNGKey(0)
        actions = jax.random.uniform(
            prng_key, (n_actions, start_state.n_dims), minval=-1, maxval=1
        )
        params = SearchParams.from_target_and_actions(
            target_state, actions, max_steps=10
        )

        curr_sphere_obs = sphere_obs.transform(
            jaxlie.SE3(
                jnp.array([*sphere_obs_handle.wxyz, *sphere_obs_handle.position])
            )
        )

        policy_output = foo(params, start_state, prng_key, coll, kin, curr_sphere_obs)

        # Also do it along the path, not just the endpoints
        # It is definitely doing tree search in a collision away manner
        # Might be also nice to _plot_ the tree it plans!
        start_state = start_state.apply_action(
            actions[policy_output.action], params.target_state, increment_steps=False
        ).value
        next_joints = start_state[0]
        # step = (next_joints - curr_joints)
        # scale = (kin.joint_vel_limit * 0.1 / (jnp.abs(step) + 1e-6)).max().clip(min=-1.0, max=1.0)
        # curr_joints = curr_joints + step * scale
        curr_joints = next_joints
        # somehow the values are also not clamped?
        # Also there's nothing in the cost fn that says that we should pick the earliest action! There needs to be some discount
        # print(curr_joints, target_joints, policy_output.action)
        urdf_vis_path.update_cfg(onp.array(curr_joints))

if __name__ == "__main__":
    main()