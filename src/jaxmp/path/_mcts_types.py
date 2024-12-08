# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
"""
Common types for formulating a tree-based path planning problem.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
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
class NodeState[T](ABC):
    value: T

    @abstractmethod
    def apply_action(
        self, action: T, rng_key: Optional[jax.Array] = None
    ) -> NodeState[T]:
        """Apply action to the current state."""
        raise NotImplementedError


@jdc.pytree_dataclass
class NodeTransition[T](ABC):
    n_actions: jdc.Static[int]
    actions: T

    @abstractmethod
    def get_transition(self, state: NodeState[T], action_idx: jax.Array) -> T:
        """Get action to be applied to the current state."""
        raise NotImplementedError

    @abstractmethod
    def get_transition_outputs(
        self, state: NodeState[T], action: T, target: NodeState[T]
    ) -> tuple[NodeState[T], jax.Array, jax.Array]:
        raise NotImplementedError

    @abstractmethod
    def get_empty_transition(self) -> T:
        """Get a do-nothing transition."""
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
        _, value, _ = transition.get_transition_outputs(
            start_state, transition.get_empty_transition(), target_state
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
            gumbel_scale=0.5,
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
        state_next, value, reward = transition.get_transition_outputs(
            state, action, target_state
        )
        prior_logits = jnp.zeros((1, transition.n_actions))
        discount = jnp.array([1.0])

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=prior_logits,
            value=value,
        )
        return recurrent_fn_output, state_next
