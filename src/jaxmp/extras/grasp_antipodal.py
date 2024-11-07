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
import trimesh.intersections


@jdc.pytree_dataclass
class AntipodalGrasps:
    centers: Float[Array, "*batch 3"]
    axes: Float[Array, "*batch 3"]

    def __len__(self) -> int:
        return self.centers.shape[0]
    
    @staticmethod
    def from_empty() -> AntipodalGrasps:
        return AntipodalGrasps(
            centers=jnp.zeros((0, 3)),
            axes=jnp.zeros((0, 3)),
        )

    @staticmethod
    def from_sample_mesh(
        mesh: trimesh.Trimesh,
        prng_key: jax.Array,
        max_samples=100,
        min_width=0.0,
        max_width=float("inf"),
        max_depth=float("inf"),
        max_angle_deviation=onp.pi / 4,
    ) -> AntipodalGrasps:
        """
        Sample antipodal grasps from a given mesh, using rejection sampling.
        May return fewer grasps than `max_samples`.
        """
        grasp_centers, grasp_axes = [], []
        dot_products = []

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
            grasp_direction = n1 - n2
            grasp_direction /= onp.linalg.norm(grasp_direction) + 1e-6

            # If the grasp width is too small, skip. (i.e., not grasping any volume).
            if grasp_width < min_width:
                continue

            # If the grasp direction is not aligned with the surface normals, skip.
            if (
                onp.dot(n1, grasp_direction) < min_dot_product
                or onp.dot(n2, -grasp_direction) < min_dot_product
            ):
                continue

            # If the grasp depth is too deep, skip.
            line_segments = cast(
                onp.ndarray,
                trimesh.intersections.mesh_plane(mesh, grasp_direction, grasp_center),
            )
            if len(line_segments) == 0:
                continue

            AB = line_segments[:, 1] - line_segments[:, 0]
            AP = grasp_center - line_segments[:, 0]
            BP = grasp_center - line_segments[:, 1]
            t = jnp.einsum("ij,ij->i", AP, AB) / jnp.einsum("ij,ij->i", AB, AB)
            dist = onp.linalg.norm(AP - t[:, None] * AB, axis=-1)
            dist = jnp.where(t < 0, onp.linalg.norm(AP, axis=-1), dist)
            dist = jnp.where(t > 1, onp.linalg.norm(BP, axis=-1), dist)

            dist = jnp.min(dist)
            if dist > max_depth:
                continue

            grasp_centers.append(grasp_center)
            grasp_axes.append(grasp_direction)
            dot_products.append(jnp.abs(onp.dot(n1, n2)))

        # Sort based on grasps' alignment with the surface normals.
        dot_products = jnp.array(dot_products)
        sort_indices = jnp.argsort(dot_products, descending=True)
        grasp_centers = jnp.array(grasp_centers)[sort_indices]
        grasp_axes = jnp.array(grasp_axes)[sort_indices]

        assert jnp.isnan(grasp_centers).sum() == 0
        assert jnp.isnan(grasp_axes).sum() == 0

        return AntipodalGrasps(
            centers=jnp.array(grasp_centers),
            axes=jnp.array(grasp_axes),
        )

    def nms(self, pos_thresh: float, angle_thresh: float) -> AntipodalGrasps:
        """
        Perform non-maximum suppression on the grasps.
        """
        centers = self.centers
        axes = self.axes
        num_centers = centers.shape[0]

        # Create a mask to keep track of points to keep
        keep = jnp.ones(num_centers, dtype=jnp.bool_)

        # Compute pairwise distances between centers
        dist_matrix = jnp.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=-1
        )
        angle_matrix = jnp.einsum("ij,kj->ik", axes, axes)

        # Vectorized approach to decide which elements to keep
        pos_mask = (dist_matrix < pos_thresh) & (
            dist_matrix > 0
        )  # Exclude zero distances (self-comparison)
        angle_mask = (angle_matrix > jnp.cos(angle_thresh)) | (
            angle_matrix < -jnp.cos(angle_thresh)
        )

        # Combine masks to determine suppression
        suppression_mask = pos_mask & angle_mask

        for i in range(num_centers):
            if not keep[i]:
                continue

            # Suppress elements in the columns of the current row that match the condition
            keep = keep & ~(suppression_mask[i])

        # Filter centers and axes using the keep mask
        return AntipodalGrasps(centers[keep], axes[keep])

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
        
        _self = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), self)
        grasp_transforms = _self.to_se3(along_axis=along_axis).as_matrix()
        for idx in range(_self.centers.shape[0]):
            if indices is not None and idx not in indices:
                continue
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(grasp_transforms[idx])
            meshes.append(mesh_copy)

        return sum(meshes, trimesh.Trimesh())

    def to_se3(
        self, along_axis: Literal["x", "y", "z"] = "x", flip_axis: bool = False
    ) -> jaxlie.SE3:
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
    random_lengths = random_lengths.at[random_test].set(
        random_lengths[random_test] - 1.0
    )
    random_lengths = jnp.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples, face_index
