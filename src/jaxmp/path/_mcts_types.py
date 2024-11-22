# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
"""
Common types for formulating a tree-based path planning problem.
"""

from __future__ import annotations
from typing import Callable, Optional, cast

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

import mctx


type RecurrentFn[T] = Callable[
    [SearchParams[T], jax.Array, jax.Array, NodeState[T]],
    tuple[mctx.RecurrentFnOutput, NodeState[T]],
]

type SearchParams[T] = tuple[NodeTransition[T], NodeState[T]]


@jdc.pytree_dataclass
class NodeState[T]:
    value: T

    @staticmethod
    def from_state(state: T) -> NodeState[T]:
        return NodeState(state)

    def apply_action(
        self, action: T, rng_key: Optional[jax.Array] = None
    ) -> NodeState[T]:
        """Apply action to the current state."""
        raise NotImplementedError


@jdc.pytree_dataclass
class NodeTransition[T]:
    n_actions: jdc.Static[int]
    actions: T

    @staticmethod
    def from_actions(n_actions: int, actions: T) -> NodeTransition[T]:
        # I need an "empty" action
        return NodeTransition(n_actions, actions)

    def get_transition(self, state: NodeState[T], action_idx: jax.Array) -> T:
        """Get action to be applied to the current state."""
        raise NotImplementedError

    def get_value(self, state: NodeState[T], action: T, target: NodeState[T]) -> jax.Array:
        """Get value of the state-action pair, `V((s, a)| target)`."""
        raise NotImplementedError

    def get_reward(
        self, state: NodeState[T], target: NodeState[T]
    ) -> jax.Array:
        """Get reward `r`."""
        raise NotImplementedError


class Search:
    @staticmethod
    @jdc.jit
    def solve[T](
        start_state: NodeState[T],
        target_state: NodeState[T],
        transition: NodeTransition[T],
        prng_key: jax.Array,
        max_depth: jdc.Static[int],
        num_simulations: jdc.Static[int],
    ) -> T:
        policy_output = Search._run_policy(
            start_state,
            target_state,
            transition,
            max_depth,
            num_simulations,
            prng_key,
        )
        _action = cast(jax.Array, policy_output.action)  # returned as chex.Array.
        action = transition.get_transition(start_state, _action)
        return action

    @staticmethod
    def _run_policy[T](
        start_state: NodeState[T],
        target_state: NodeState[T],
        transition: NodeTransition[T],
        max_depth: int,
        num_simulations: int,
        prng_key: jax.Array,
    ) -> mctx.PolicyOutput:
        value = transition.get_value(
            start_state, target_state
        )
        root = mctx.RootFnOutput(
            prior_logits=jnp.zeros([1, transition.n_actions]),
            value=value,
            embedding=start_state,
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=(transition, target_state),
            rng_key=prng_key,
            root=root,
            recurrent_fn=Search._recurrent_fn,
            num_simulations=num_simulations,
            max_depth=max_depth,
        )
        return policy_output

    @staticmethod
    def _recurrent_fn[T](
        search_params: SearchParams[T],
        rng_key: jax.Array,
        action_idx: jax.Array,
        state: NodeState[T],
    ) -> tuple[mctx.RecurrentFnOutput, NodeState[T]]:
        transition, target_state = search_params
        action = transition.get_transition(state, action_idx)
        state_next = state.apply_action(action, rng_key)

        value = transition.get_value(state, action, target_state)
        reward = transition.get_reward(state, state_next, target_state)
        prior_logits = jnp.zeros((1, transition.n_actions))
        discount = jnp.array([0.0])

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=prior_logits,
            value=value,
        )
        return recurrent_fn_output, state_next
