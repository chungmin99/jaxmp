"""
Common types to formulate path planning as a MCTS problem.
"""
# collision

from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

import mctx


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

    def get_recurrent_fn(
        self, dist_heuristc_fn: Callable[[SearchState, SearchState], jax.Array]
    ):
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
            success = jnp.all(jnp.isclose(state_next.value, params.target_state.value, atol=1e-3))

            prev_value = dist_heuristc_fn(state, params.target_state)
            next_value = dist_heuristc_fn(state_next, params.target_state)

            value = prev_value
            reward = next_value - prev_value + success * 100

            discount = jnp.where(
                jnp.logical_or(
                    state_next.n_steps > params.max_steps,
                    success,
                ), 1.0, 0.0
            )

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
