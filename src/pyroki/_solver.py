"""
Solver for robot kinematic optimization problems, by wrapping `jaxls`.
"""

from __future__ import annotations

from typing import Literal, Optional

import jax
import jax_dataclasses as jdc

import jaxls


def solve(
    vars: list[jaxls.Var],
    factors: list[CostFactor],
    init_vars: list,
    *,
    solver_type: Literal[
        "cholmod", "conjugate_gradient", "dense_cholesky"
    ] = "conjugate_gradient",
    max_iterations: int = 100,
    verbose: bool = False,
) -> jaxls.VarValues:
    """Solve the robot kinematic optimization problem."""
    factors_jaxls = [cf._make_factor() for cf in factors]
    if len(init_vars) != len(vars):
        if len(init_vars) == 0:
            init_vars = vars
        else:
            raise ValueError(
                f"Number of initial variables ({len(init_vars)}) must match number of variables ({len(vars)})."
            )

    graph = jaxls.FactorGraph.make(factors_jaxls, vars, use_onp=False)
    solution = graph.solve(
        linear_solver=solver_type,
        initial_vals=jaxls.VarValues.make(init_vars),
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5,
            parameter_tolerance=1e-5,
            max_iterations=max_iterations,
        ),
        verbose=verbose,
    )
    return solution


@jdc.pytree_dataclass
class CostFactor[*Args]:
    """Cost function."""

    cost_inputs: tuple[*Args]
    weights: Optional[jax.Array | float] = None

    @classmethod
    def make(
        cls, *cost_inputs: *Args, weights: Optional[jax.Array] = None
    ) -> CostFactor[*Args]:
        """
        Factory method for creating a cost factor using positional arguments.

        This allows cleaner construction of cost terms, avoiding the need to
        explicitly unpack a tuple into the dataclass.

        Example:
            cost = MyCost.make(robot, joint_var, target_pose, weights=jnp.array([1.0]))
        """
        factor = cls(cost_inputs=cost_inputs, weights=weights)
        return factor

    def cost_fn(self, vals: jaxls.VarValues, *args: *Args) -> jax.Array:
        raise NotImplementedError

    def _make_factor(self) -> jaxls.Factor:
        """
        Make a factor from the cost function.
        There must be at least one variable `jaxls.Var` provided in the arguments.
        """

        assert len(self.cost_inputs) > 0
        assert any(isinstance(arg, jaxls.Var) for arg in self.cost_inputs)

        # Wrapper cost function around `jaxls`, to avoid exposing `jaxls.VarValues`, and the `vals[var]` syntax.
        def cost_fn(vals: jaxls.VarValues, *args: *Args) -> jax.Array:
            residual = self.cost_fn(vals, *args)

            if self.weights is not None:
                residual = residual * self.weights

            # Flatten the residual; `jaxls` expects a 1D array.
            residual = residual.flatten()

            return residual

        cost_fn.__name__ = f"{self.__class__.__name__}"
        return jaxls.Factor(cost_fn, self.cost_inputs)
