"""
Antipodal grasp sampling.
"""

from __future__ import annotations

from typing import Literal, Optional, cast
import jax
import numpy as onp
from jax import Array
import jax.numpy as jnp
from jaxtyping import Float
import jax_dataclasses as jdc
import jaxlie

import trimesh
import trimesh.sample


@jdc.pytree_dataclass
class AntipodalGrasps:
    centers: Float[Array, "*batch 3"]
    axes: Float[Array, "*batch 3"]

    def __len__(self) -> int:
        return self.centers.shape[0]

    @staticmethod
    def from_sample_mesh(
        mesh: trimesh.Trimesh,
        prng_key: jax.Array,
        max_samples=100,
        max_width=float("inf"),
        max_angle_deviation=onp.pi / 4,
    ) -> AntipodalGrasps:
        """
        Sample antipodal grasps from a given mesh, using rejection sampling.
        May return fewer grasps than `max_samples`.
        """
        grasp_centers, grasp_axes = [], []

        sampled_points, sampled_face_indices = cast(
            tuple[onp.ndarray, onp.ndarray],
            _sample_surface(mesh, max_samples, prng_key),
        )
        min_dot_product = onp.cos(max_angle_deviation)

        for sample_idx in range(max_samples):
            p1 = sampled_points[sample_idx]
            n1 = mesh.face_normals[sampled_face_indices[sample_idx]]

            # Raycast!
            locations, _, index_tri = mesh.ray.intersects_location(
                p1.reshape(1, 3), -n1.reshape(1, 3), multiple_hits=False
            )

            if len(locations) == 0:
                continue

            p2 = locations[0]
            n2 = mesh.face_normals[index_tri[0]]

            # Check grasp width.
            grasp_width = onp.linalg.norm(p2 - p1)
            if grasp_width > max_width:
                continue

            # Check for antipodal condition.
            grasp_center = (p1 + p2) / 2
            grasp_direction = p1 - p2
            grasp_direction /= onp.linalg.norm(grasp_direction)

            if (
                onp.dot(n1, grasp_direction) > min_dot_product
                and onp.dot(n2, -grasp_direction) > min_dot_product
            ):
                grasp_centers.append(grasp_center)
                grasp_axes.append(grasp_direction)

        return AntipodalGrasps(
            centers=jnp.array(grasp_centers),
            axes=jnp.array(grasp_axes),
        )

    def to_trimesh(
        self,
        axes_radius: float = 0.005,
        axes_height: float = 0.1,
        indices: Optional[tuple[int, ...]] = None,
        along_axis: Literal["x", "y", "z"] = "x",
    ) -> trimesh.Trimesh:
        """
        Convert the grasp to a trimesh object.
        """
        # Create "base" grasp visualization, centered at origin + lined up with x-axis.
        transform = onp.eye(4)
        if along_axis == "x":
            rotation = trimesh.transformations.rotation_matrix(onp.pi / 2, [0, 1, 0])
        elif along_axis == "y":
            rotation = trimesh.transformations.rotation_matrix(onp.pi / 2, [1, 0, 0])
        else:
            rotation = onp.eye(4)

        transform[:3, :3] = rotation[:3, :3]
        mesh = trimesh.creation.cylinder(
            radius=axes_radius, height=axes_height, transform=transform
        )
        mesh.visual.vertex_colors = [150, 150, 255, 255]  # type: ignore[attr-defined]

        meshes = []
        grasp_transforms = self.to_se3(along_axis=along_axis).as_matrix()
        for idx in range(self.centers.shape[0]):
            if indices is not None and idx not in indices:
                continue
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(grasp_transforms[idx])
            meshes.append(mesh_copy)

        return sum(meshes, trimesh.Trimesh())

    def to_se3(self, along_axis: Literal["x", "y", "z"] = "x", flip_axis: bool = False) -> jaxlie.SE3:
        # Create rotmat, first assuming the x-axis is the grasp axis.
        x_axes = self.axes
        if flip_axis:
            x_axes = -x_axes

        delta = x_axes + (
            jnp.sign(x_axes[..., 0] + 1e-6)[..., None]
            * jnp.roll(x_axes, shift=1, axis=-1)
        )
        y_axes = jnp.cross(x_axes, delta)
        y_axes = y_axes / (jnp.linalg.norm(y_axes, axis=-1, keepdims=True) + 1e-6)
        assert jnp.isclose(x_axes, y_axes).all(axis=-1).sum() == 0

        z_axes = jnp.cross(x_axes, y_axes)

        if along_axis == "x":
            rotmat = jnp.stack([x_axes, y_axes, z_axes], axis=-1)
        elif along_axis == "y":
            rotmat = jnp.stack([z_axes, x_axes, y_axes], axis=-1)
        elif along_axis == "z":
            rotmat = jnp.stack([y_axes, z_axes, x_axes], axis=-1)

        assert jnp.isnan(rotmat).sum() == 0

        # Use the axis-angle representation to create the rotation matrix.
        return jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(rotmat), self.centers
        )


# Copied from `trimesh.sample.sample_surface`, but using jax PRNG.
def _sample_surface(
    mesh,
    count: int,
    prng_key: jax.Array,
):
    """
    Sample the surface of a mesh, returning the specified
    number of points

    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    prng_key: jax.Array
        PRNG key

    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """

    face_weight = mesh.area_faces

    # cumulative sum of weights (len(mesh.faces))
    weight_cum = jnp.cumsum(face_weight)

    # seed the random number generator as requested
    random_values = jax.random.uniform(prng_key, (count,))

    # last value of cumulative sum is total summed weight/area
    face_pick = random_values * weight_cum[-1]
    # get the index of the selected faces
    face_index = jnp.searchsorted(weight_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.vertices[mesh.faces[:, 0]]
    tri_vectors = mesh.vertices[mesh.faces[:, 1:]].copy()
    tri_vectors -= jnp.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = jax.random.uniform(prng_key, (len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths = random_lengths.at[random_test].set(random_lengths[random_test] - 1.0)
    random_lengths = jnp.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples, face_index