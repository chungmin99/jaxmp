from __future__ import annotations

from typing import Callable, Dict, Tuple, cast

import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float, Array

from ._geometry import CollGeom, HalfSpace, Sphere, Capsule, Box
from ._geometry_pairs import (
    halfspace_sphere,
    halfspace_capsule,
    halfspace_box,
    sphere_sphere,
    sphere_capsule,
    sphere_box,
    capsule_capsule,
    capsule_box,
)

COLLISION_FUNCTIONS: Dict[
    Tuple[type[CollGeom], type[CollGeom]], Callable[..., Float[Array, "*batch"]]
] = {
    # Reference the imported functions
    (HalfSpace, Sphere): halfspace_sphere,
    (HalfSpace, Capsule): halfspace_capsule,
    (HalfSpace, Box): halfspace_box,
    (Sphere, Sphere): sphere_sphere,
    (Sphere, Capsule): sphere_capsule,
    (Sphere, Box): sphere_box,
    (Capsule, Capsule): capsule_capsule,
    (Capsule, Box): capsule_box,
}


def _get_coll_func(
    geom1_cls: type[CollGeom], geom2_cls: type[CollGeom]
) -> Callable[[CollGeom, CollGeom], Float[Array, "*batch"]]:
    """Get appropriate collision function (distance only) for given geometry types."""
    func = COLLISION_FUNCTIONS.get((geom1_cls, geom2_cls))
    if func is not None:
        return cast(Callable[[CollGeom, CollGeom], Float[Array, "*batch"]], func)

    func_swapped = COLLISION_FUNCTIONS.get((geom2_cls, geom1_cls))
    if func_swapped is not None:
        return cast(
            Callable[[CollGeom, CollGeom], Float[Array, "*batch"]],
            lambda g1, g2: func_swapped(g2, g1),
        )

    raise NotImplementedError(
        f"No collision function found for {geom1_cls.__name__} and {geom2_cls.__name__}"
    )


@jdc.jit
def collide(geom1: CollGeom, geom2: CollGeom) -> Float[Array, "*batch"] | None:
    """Calculate collision distance between two geometric objects, handling broadcasting."""
    try:
        broadcast_shape = jnp.broadcast_shapes(
            geom1.get_batch_axes(), geom2.get_batch_axes()
        )
    except ValueError as e:
        raise ValueError(
            f"Cannot broadcast geometry shapes {geom1.get_batch_axes()} and {geom2.get_batch_axes()}"
        ) from e

    geom1_b = geom1.broadcast_to(*broadcast_shape)
    geom2_b = geom2.broadcast_to(*broadcast_shape)

    geom1_cls = type(geom1)
    geom2_cls = type(geom2)

    try:
        func = _get_coll_func(geom1_cls, geom2_cls)
    except NotImplementedError:
        return jnp.full(broadcast_shape, jnp.nan)

    dist_result = func(geom1_b, geom2_b)

    return dist_result
