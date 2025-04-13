"""
Solver for robot kinematic optimization problems, using the `pyroki.optim` module.
"""

from __future__ import annotations

from typing import Literal, Optional, Callable, Hashable

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
) -> tuple[jaxls.VarValues, jax.Array]:
    """Solve the robot kinematic optimization problem.

    Returns:
        - solution: The optimized variable values.
        - residuals: The residuals of the cost factors.
    """
    factors_jaxls = [cf._make_factor() for cf in factors]
    if len(init_vars) != len(vars):
        if len(init_vars) == 0:
            init_vars = vars
        else:
            raise ValueError(
                f"Number of initial variables ({len(init_vars)}) must "
                f"match number of variables ({len(vars)})."
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
    cost_vector = graph.compute_residual_vector(solution)
    return solution, cost_vector


@jdc.pytree_dataclass
class CostFactor[*Args]:
    """Cost function."""

    cost_inputs: tuple[*Args]
    weights: jax.Array | float

    @classmethod
    def make(
        cls,
        *cost_inputs: *Args,
        weights: Optional[jax.Array | float] = None,
    ) -> CostFactor[*Args]:
        """
        Factory method for creating a cost factor using positional arguments.

        This allows cleaner construction of cost terms, avoiding the need to
        explicitly unpack a tuple into the dataclass.

        Example:
            cost = MyCost.make(robot, joint_var, target_pose, weights=jnp.array([1.0]))
        """
        if weights is None:
            weights = 1.0
        factor = cls(cost_inputs=cost_inputs, weights=weights)
        return factor

    def cost_fn(self, vals: jaxls.VarValues, *args: *Args) -> jax.Array:
        raise NotImplementedError

    def _make_factor(self) -> jaxls.Factor:
        """
        Make a factor from the cost function.
        There must be at least one variable `optim.Var` provided in the arguments.
        """

        assert len(self.cost_inputs) > 0
        assert any(isinstance(arg, jaxls.Var) for arg in self.cost_inputs)

        # Add weights to the cost inputs if provided.
        return jaxls.Factor(
            self._cost_fn,
            self.cost_inputs,
            signature_fn=self._signature_fn,
            name=self.__class__.__name__,
        )

    def _cost_fn(self, vals: jaxls.VarValues, *args: *Args) -> jax.Array:
        """
        Wrapper cost function to apply weights and flatten the residual.
        """
        residual = self.cost_fn(vals, *args)

        if self.weights is not None:
            residual = residual * self.weights

        # Flatten the residual.
        residual = residual.flatten()

        return residual

    def _signature_fn(self, _: Callable) -> Hashable:
        """
        Retrieve the fields that influence the cost function, including:
        - the cost function itself,
        - class properties accessible via `self`.

        Exclude the `cost_inputs` field, as it is passed directly
        to the cost function as arguments.
        """

        keys = list(self.__dataclass_fields__.keys())
        keys.remove("cost_inputs")

        # Determine the field as unique based on its ID / memory address.
        id_list = [id(getattr(self, key)) for key in keys]
        return (self.__class__, *id_list)
