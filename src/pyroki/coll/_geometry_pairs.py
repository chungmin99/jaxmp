from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float, Array

from ._geometry import HalfSpace, Sphere, Capsule, Box
from . import _utils


# --- HalfSpace Collision Implementations ---


def _halfspace_sphere_dist(
    halfspace_normal: Float[Array, "*batch 3"],
    halfspace_point: Float[Array, "*batch 3"],
    sphere_pos: Float[Array, "*batch 3"],
    sphere_radius: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between a halfspace boundary plane and sphere center, minus radius."""
    dist = (
        jnp.einsum("...i,...i->...", sphere_pos - halfspace_point, halfspace_normal)
        - sphere_radius
    )
    return dist


def halfspace_sphere(halfspace: HalfSpace, sphere: Sphere) -> Float[Array, "*batch"]:
    """Calculates distance between a halfspace and a sphere."""
    dist = _halfspace_sphere_dist(
        halfspace.normal,
        halfspace.pose.translation(),
        sphere.pose.translation(),
        sphere.radius,
    )
    return dist


def halfspace_capsule(halfspace: HalfSpace, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculates distance between halfspace and capsule (closest end)."""
    halfspace_normal = halfspace.normal
    halfspace_point = halfspace.pose.translation()
    cap_center = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.length[..., None]
    dist1 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center + segment_offset, cap_radius
    )
    dist2 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center - segment_offset, cap_radius
    )
    final_dist = jnp.minimum(dist1, dist2)
    return final_dist


def halfspace_box(halfspace: HalfSpace, box: Box) -> Float[Array, "*batch"]:
    """Calculates distance between halfspace and box."""
    halfspace_normal = halfspace.normal
    halfspace_point = halfspace.pose.translation()

    box_pose = box.pose
    box_extents = box.extents
    half_extents = box_extents / 2.0

    local_vertices = (
        jnp.array(
            [
                [-1, -1, -1],
                [-1, -1, +1],
                [-1, +1, -1],
                [-1, +1, +1],
                [+1, -1, -1],
                [+1, -1, +1],
                [+1, +1, -1],
                [+1, +1, +1],
            ]
        )
        * half_extents[..., None, :]
    )
    world_vertices = box_pose.apply(local_vertices)
    hs_normal_b = halfspace_normal[..., None, :]
    hs_point_b = halfspace_point[..., None, :]
    vertex_distances = jnp.einsum(
        "...vi,...vi->...v", world_vertices - hs_point_b, hs_normal_b
    )
    min_vertex_distance = jnp.min(vertex_distances, axis=-1)
    return min_vertex_distance


# --- Sphere/Capsule/Box Collision Implementations ---


def _sphere_sphere_dist(
    pos1: Float[Array, "*batch 3"],
    radius1: Float[Array, "*batch"],
    pos2: Float[Array, "*batch 3"],
    radius2: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between two spheres."""
    _, dist_center = _utils.normalize_with_norm(pos2 - pos1)
    dist = dist_center - (radius1 + radius2)
    return dist


def sphere_sphere(sphere1: Sphere, sphere2: Sphere) -> Float[Array, "*batch"]:
    """Calculate distance between two spheres."""
    dist = _sphere_sphere_dist(
        sphere1.pose.translation(),
        sphere1.radius,
        sphere2.pose.translation(),
        sphere2.radius,
    )
    return dist


def sphere_capsule(sphere: Sphere, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between sphere and capsule."""
    cap_pos = capsule.pose.translation()
    sphere_pos = sphere.pose.translation()
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.length[..., None]
    cap_a = cap_pos - segment_offset
    cap_b = cap_pos + segment_offset
    pt_on_axis = _utils.closest_segment_point(cap_a, cap_b, sphere_pos)
    dist = _sphere_sphere_dist(sphere_pos, sphere.radius, pt_on_axis, capsule.radius)
    return dist


def sphere_box(sphere: Sphere, box: Box) -> Float[Array, "*batch"]:
    """Calculate distance between sphere and box."""
    sphere_center_world = sphere.pose.translation()
    sphere_radius = sphere.radius
    box_pose = box.pose
    box_half_extents = box.extents / 2.0
    sphere_center_local = box_pose.inverse().apply(sphere_center_world)
    closest_point_local = jnp.clip(
        sphere_center_local, -box_half_extents, box_half_extents
    )
    closest_point_world = box_pose.apply(closest_point_local)
    dist_center_to_box = jnp.linalg.norm(
        sphere_center_world - closest_point_world, axis=-1
    )
    dist = dist_center_to_box - sphere_radius
    return dist


def capsule_capsule(capsule1: Capsule, capsule2: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between two capsules."""
    pos1 = capsule1.pose.translation()
    axis1 = capsule1.axis
    length1 = capsule1.length
    radius1 = capsule1.radius
    segment1_offset = axis1 * length1[..., None]
    a1 = pos1 - segment1_offset
    b1 = pos1 + segment1_offset

    pos2 = capsule2.pose.translation()
    axis2 = capsule2.axis
    length2 = capsule2.length
    radius2 = capsule2.radius
    segment2_offset = axis2 * length2[..., None]
    a2 = pos2 - segment2_offset
    b2 = pos2 + segment2_offset

    pt1_on_axis, pt2_on_axis = _utils.closest_segment_to_segment_points(a1, b1, a2, b2)
    dist = _sphere_sphere_dist(pt1_on_axis, radius1, pt2_on_axis, radius2)
    return dist


def capsule_box(capsule: Capsule, box: Box) -> Float[Array, "*batch"]:
    """Calculate approximate distance between capsule and box."""
    cap_center_world = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis = capsule.axis
    cap_length = capsule.length  # Half-length
    box_pose = box.pose
    box_half_extents = box.extents / 2.0
    cap_center_local = box_pose.inverse().apply(cap_center_world)
    closest_point_on_box_local = jnp.clip(
        cap_center_local, -box_half_extents, box_half_extents
    )
    closest_point_on_box_world = box_pose.apply(closest_point_on_box_local)
    segment_offset = cap_axis * cap_length[..., None]
    cap_a = cap_center_world - segment_offset
    cap_b = cap_center_world + segment_offset
    pt_on_axis = _utils.closest_segment_point(cap_a, cap_b, closest_point_on_box_world)
    dist_axis_to_box = jnp.linalg.norm(closest_point_on_box_world - pt_on_axis, axis=-1)
    dist = dist_axis_to_box - cap_radius
    return dist
