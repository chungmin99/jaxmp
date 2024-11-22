from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

import mctx
from jaxmp import JaxKinTree
from jaxmp.extras import load_urdf


@jdc.pytree_dataclass
class SearchParams:
    target_state: SearchState
    max_steps: jdc.Static[int]

    actions: jax.Array
    n_actions: jdc.Static[int]

    @staticmethod
    def from_target_and_actions(
        target: SearchState, actions: jax.Array, max_steps: int
    ) -> SearchParams:
        n_actions, n_state_dim = actions.shape
        assert n_state_dim == target.n_dims
        return SearchParams(target, max_steps, actions, n_actions)

    def get_recurrent_fn(self):
        def recurrent_fn(
            params: SearchParams,
            rng_key: jax.Array,
            action_idx: jax.Array,
            state: SearchState,
        ):
            del rng_key

            state_next = state.apply_action(
                params.actions[action_idx], params.target_state
            )
            value = state_next.get_dist_heuristic(params.target_state)
            reward = value
            discount = jnp.where(state_next.n_steps > params.max_steps, 1.0, 0.0)

            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=jnp.zeros((1, params.n_actions)),
                value=value,
            )
            return recurrent_fn_output, state_next

        return recurrent_fn


@jdc.pytree_dataclass
class SearchState:
    value: jax.Array
    n_dims: jdc.Static[int]
    n_steps: jax.Array

    @staticmethod
    def from_value(value: jax.Array):
        return SearchState(value=value, n_dims=value.shape[-1], n_steps=jnp.array([0]))

    def get_dist_heuristic(self, target: SearchState):
        # By default, we use L2 distance.
        value = -jnp.linalg.norm(self.value - target.value, axis=-1)
        return value

    def apply_action(
        self, action: jax.Array, target: SearchState, increment_steps=True
    ):
        curr_dist_to_target = jnp.abs(target.value - self.value)
        action = action * curr_dist_to_target

        with jdc.copy_and_mutate(self) as state_next:
            state_next.value = self.value + action
            if increment_steps:
                state_next.n_steps = self.n_steps + 1
        return state_next


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


def main():
    n_dims = 2
    max_steps = 10
    n_actions = 100
    prng_key_val = 42

    urdf = load_urdf("ur5")
    kin = JaxKinTree.from_urdf(urdf)
    n_dims = kin.num_actuated_joints

    prng_key = jax.random.PRNGKey(prng_key_val)

    start_state = SearchState.from_value(jnp.full([1, n_dims], 0, dtype=jnp.float32))
    # target_state = SearchState.from_value(jnp.full([1, n_dims], 1, dtype=jnp.float32))
    # [-0.16674091 -0.7618493   1.6530751   3.7709625  -1.5622753  -0.16651219]
    target_state = SearchState.from_value(
        jnp.array(
            [[-0.16674091, -0.7618493, 1.6530751, 3.7709625, -1.5622753, -0.16651219]]
        )
    )
    assert start_state.n_dims == target_state.n_dims
    actions = jax.random.uniform(
        prng_key, (n_actions, start_state.n_dims), minval=-1, maxval=1
    )

    params = SearchParams.from_target_and_actions(
        target_state, actions, max_steps=max_steps
    )

    print("actions")
    print(actions)
    print()

    print("before", start_state.value)
    print("target", target_state.value)
    print()

    for n_step in range(100):
        print("Step", n_step)
        policy_output = step(params, start_state, prng_key)
        print("\taction", policy_output.action)
        start_state = start_state.apply_action(
            actions[policy_output.action], params.target_state, increment_steps=False
        )
        print("\tafter ", start_state.value)


main()


# @jdc.pytree_dataclass
# class JointState:
#     joints: jax.Array
#     num_joints: jdc.Static[int]
#     num_steps: jax.Array

#     @staticmethod
#     def from_joints(joints: jax.Array):
#         return JointState(
#             joints=joints, num_joints=joints.shape[-1], num_steps=jnp.array([0])
#         )


# @jdc.pytree_dataclass
# class Params:
#     target_joints: JointState
#     num_actions: jdc.Static[int] = 3**2
#     max_steps: jdc.Static[int] = 3

#     # something about actions
#     def value(self, state: JointState):
#         dist = jnp.abs(self.target_joints.joints - state.joints).sum()
#         return -dist  # lower dist -> higher reward


# def action_id_to_action(act_id: jax.Array, num_actions: int, num_joints: int) -> jax.Array:
#     # return act_id / num_actions - 0.5
#     return jnp.array([
#         (act_id // (num_joints+1)) / num_joints - 0.5,
#         (act_id % (num_joints+1)) / num_joints - 0.5,
#     ]).T

# print("actions:")
# for i in range(9):
#     print(action_id_to_action(i, 9, 2))

# def recurrent_fn(
#     params: Params, rng_key: jax.Array, action: jax.Array, state: JointState
# ):
#     # action _is_ important (or you don't know where to step next)
#     action = action_id_to_action(action, params.num_actions, state.num_joints)

#     # actions = jax.random.uniform(rng_key, (100, 2))
#     # or basically, ignore the action.
#     # (lol oops) (just using mctx as a tree really, we should be able to look at the tree to get the full traj.)
#     # We can try to do a ait* or bit* thing here!
#     # And do an internal parallel steering here.
#     # but maybe it's useful for a training thing down the road.
#     # action = jax.random.uniform(rng_key, (1, 2), minval=-1, maxval=1) * 0.1

#     with jdc.copy_and_mutate(state) as state_next:
#         state_next.joints = state.joints + action
#         state_next.num_steps = state.num_steps + 1

#     value = params.value(state_next).reshape(1)
#     discount = jnp.where(state_next.num_steps > params.max_steps, 1.0, 0.0)
#     # reward = jnp.where(jnp.abs(value) < 0.01, 1.0, 0.0).reshape(1)
#     reward = value

#     recurrent_fn_output = mctx.RecurrentFnOutput(
#         reward=reward,
#         discount=discount,
#         prior_logits=jnp.zeros((1, params.num_actions)),
#         value=value,
#     )
#     return recurrent_fn_output, state_next


# start_state = JointState.from_joints(jnp.zeros([1, 2]))
# params = Params(target_joints=JointState.from_joints(jnp.ones([1, 2]) * 2))
# value = params.value(start_state).reshape(1)

# root = mctx.RootFnOutput(
#     prior_logits=jnp.zeros([1, params.num_actions]),
#     value=value,
#     embedding=start_state,
# )

# rng_key = jax.random.PRNGKey(0)
# policy_output = mctx.gumbel_muzero_policy(
#     params=params,
#     rng_key=rng_key,
#     root=root,
#     recurrent_fn=recurrent_fn,
#     num_simulations=100,
#     max_depth=params.max_steps,
# )
# print(start_state.joints)
# print(policy_output.action)

# # breakpoint()

# for _ in range(10):
#     action = action_id_to_action(
#         policy_output.action, params.num_actions, root.embedding.num_joints
#     )
#     with jdc.copy_and_mutate(root.embedding) as embedding_next:
#         embedding_next.joints = root.embedding.joints + action
#     print("action", action)
#     print("next", embedding_next.joints)
#     value = params.value(embedding_next).reshape(1)
#     root = mctx.RootFnOutput(
#         prior_logits=jnp.zeros([1, params.num_actions]),
#         value=value,
#         embedding=embedding_next,
#     )
#     policy_output = mctx.gumbel_muzero_policy(
#         params=params,
#         rng_key=rng_key,
#         root=root,
#         recurrent_fn=recurrent_fn,
#         num_simulations=100,
#         max_depth=params.max_steps,
#     )
#     # print(policy_output.action)

#     batch_index = 0
#     selected_action = policy_output.action[batch_index]
#     q_value = policy_output.search_tree.summary().qvalues[batch_index, selected_action]
#     # print(policy_output.search_tree.summary().qvalues)
#     print("Selected action:", selected_action)
#     # To estimate the value of the root state, use the Q-value of the selected
#     # action. The Q-value is not affected by the exploration at the root node.
#     print("Selected action Q-value:", q_value)


# with jdc.copy_and_mutate(root.embedding) as embedding_next:
#     embedding_next.joints = root.embedding.joints + action_id_to_action(
#         policy_output.action, params.num_actions, root.embedding.num_joints
#     )
# print(embedding_next.joints)
