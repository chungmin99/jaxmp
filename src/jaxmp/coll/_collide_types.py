"""
Small wrapper around mjx collision functions to handle batched geometries,
This avoids dealing with the MJX model and data structures directly.

Uses the MJX private API, may break in future versions.
"""

from __future__ import annotations

import abc
from typing import cast

import trimesh
import scipy
import jax
import numpy as onp
import jax.numpy as jnp
import jaxlie
from jaxtyping import Float
import jax_dataclasses as jdc

from mujoco.mjx._src.types import ConvexMesh
from mujoco.mjx._src.mesh import _get_face_norm, _get_edge_normals


@jdc.pytree_dataclass
class CollGeom(abc.ABC):
    pose: Float[jaxlie.SE3, "*batch 7"]  # SE3.
    size: Float[jax.Array, "*batch 3"]  # Object shape (e.g., radii, height).

    @property
    def pos(self):
        return self.pose.translation()

    @property
    def mat(self):
        return self.pose.rotation().as_matrix()

    def get_batch_axes(self):
        return self.pose.get_batch_axes()

    def broadcast_to(self, *shape):
        with jdc.copy_and_mutate(self, validate=False) as _self:
            _self.pose = jaxlie.SE3(jnp.broadcast_to(_self.pose.wxyz_xyz, (*shape, 7)))
            _self.size = jnp.broadcast_to(_self.size, shape + (3,))
        return _self

    def reshape(self, *shape):
        with jdc.copy_and_mutate(self, validate=False) as _self:
            _self.pose = jaxlie.SE3(_self.pose.wxyz_xyz.reshape(shape + (7,)))
            _self.size = _self.size.reshape(shape + (3,))
        return _self

    def transform(self, tf: jaxlie.SE3):
        with jdc.copy_and_mutate(self, validate=False) as _self:
            _self.pose = tf @ _self.pose
        return _self

    def to_trimesh(self) -> trimesh.Trimesh:
        _self = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), self)

        meshes = [trimesh.Trimesh()]
        for i in range(_self.get_batch_axes()[0]):
            meshes.append(
                self._create_one_mesh(_self.pos[i], _self.mat[i], _self.size[i])
            )

        return cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))

    @abc.abstractmethod
    def _create_one_mesh(
        self,
        pos: Float[jax.Array, "3"],
        mat: Float[jax.Array, "3 3"],
        size: Float[jax.Array, "3"],
    ):
        raise NotImplementedError


@jdc.pytree_dataclass
class Plane(CollGeom):
    @staticmethod
    def from_point_and_normal(point: jax.Array, normal: jax.Array) -> Plane:
        batch_axes = point.shape[:-1]
        assert point.shape[-1] == 3

        mat = Plane._normal_to_SO3(normal)
        assert mat.shape[:-2] == batch_axes

        size = jnp.zeros(batch_axes + (3,))
        pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(mat),  # SO3
            point,
        )
        return Plane(pose=pose, size=size)

    @staticmethod
    def _normal_to_SO3(normal: jax.Array) -> jax.Array:
        # Align z-axis with normal.
        delta = normal + (
            jnp.sign(normal[..., 0] + 1e-6)[..., None]
            * jnp.roll(normal, shift=1, axis=-1)
        )
        x_axes = jnp.cross(normal, delta)
        x_axes = x_axes / (jnp.linalg.norm(x_axes, axis=-1, keepdims=True) + 1e-6)
        assert jnp.isclose(normal, x_axes).all(axis=-1).sum() == 0
        y_axes = jnp.cross(normal, x_axes)
        return jnp.stack([x_axes, y_axes, normal], axis=-1)

    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        plane = trimesh.creation.box(extents=[5, 5, 0.001])
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        plane.vertices = trimesh.transform_points(plane.vertices, tf)
        return plane


@jdc.pytree_dataclass
class Sphere(CollGeom):
    @staticmethod
    def from_center_and_radius(center: jax.Array, radius: jax.Array) -> Sphere:
        batch_axes = center.shape[:-1]
        assert center.shape[-1] == 3

        mat = jaxlie.SO3.identity(batch_axes).as_matrix()

        # Uses sphere.size[0] as the radius.
        assert radius.shape == batch_axes + (1,)

        size = jnp.zeros(batch_axes + (2,))
        size = jnp.concatenate([radius, size], axis=-1)
        pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(mat),  # SO3
            center,
        )
        return Sphere(pose=pose, size=size)

    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        sphere = trimesh.creation.icosphere(radius=size[0].item())
        tf = onp.eye(4)
        tf[:3, 3] = pos
        sphere.vertices = trimesh.transform_points(sphere.vertices, tf)
        return sphere


@jdc.pytree_dataclass
class Capsule(CollGeom):
    @staticmethod
    def from_radius_and_height(
        radius: jax.Array, height: jax.Array, transform: jaxlie.SE3
    ) -> Capsule:
        batch_axes = transform.get_batch_axes()

        # Uses cylinder.size[0] as the radius and cylinder.size[1] as the height.
        segment = height / 2
        shape = jnp.concatenate([radius, segment, jnp.zeros_like(radius)], axis=-1)
        shape = jnp.broadcast_to(shape, batch_axes + (3,))

        return Capsule(pose=transform, size=shape)

    @staticmethod
    def from_min_cylinder(mesh: trimesh.Trimesh) -> Capsule:
        """
        Approximate a minimum bounding capsule for a mesh using a minimum cylinder.
        """
        import trimesh.bounds

        results = trimesh.bounds.minimum_cylinder(mesh)

        assert "transform" in results
        assert "radius" in results
        assert "height" in results

        tf_mat = results["transform"]
        radius = results["radius"]
        height = results["height"]
        tf = jaxlie.SE3.from_matrix(tf_mat)

        cap = Capsule.from_radius_and_height(
            radius=jnp.array([radius]),
            height=jnp.array([height]),
            transform=tf,
        )

        return cap

    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        capsule = trimesh.creation.capsule(
            radius=size[0].item(), height=size[1].item() * 2
        )
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        capsule.vertices = trimesh.transform_points(capsule.vertices, tf)
        return capsule


@jdc.pytree_dataclass
class Ellipsoid(CollGeom):
    @staticmethod
    def from_center_and_abc(center: jax.Array, abc: jax.Array) -> Ellipsoid:
        batch_axes = center.shape[:-1]
        assert center.shape[-1] == 3

        mat = jaxlie.SO3.identity(batch_axes).as_matrix()

        # Uses ellipsoid.size as the radii.
        assert abc.shape == batch_axes + (3,)
        pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(mat),  # SO3
            center,
        )
        return Ellipsoid(pose=pose, size=abc)

    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        ellipsoid = trimesh.creation.icosphere(radius=size[0].item())
        ellipsoid.apply_scale(size)
        tf = onp.eye(4)
        tf[:3, 3] = pos
        ellipsoid.vertices = trimesh.transform_points(ellipsoid.vertices, tf)
        return ellipsoid


@jdc.pytree_dataclass
class Cylinder(CollGeom):
    @staticmethod
    def from_radius_and_height(
        radius: jax.Array, height: jax.Array, transform: jaxlie.SE3
    ) -> Cylinder:
        batch_axes = transform.get_batch_axes()

        # Uses cylinder.size[0] as the radius and cylinder.size[1] as the height.
        segment = height / 2
        shape = jnp.concatenate([radius, segment, jnp.zeros_like(radius)], axis=-1)
        shape = jnp.broadcast_to(shape, batch_axes + (3,))

        return Cylinder(pose=transform, size=shape)

    @staticmethod
    def from_min_cylinder(mesh: trimesh.Trimesh) -> Cylinder:
        """
        Approximate a minimum bounding cylinder for a mesh.
        """
        import trimesh.bounds

        results = trimesh.bounds.minimum_cylinder(mesh)

        assert "transform" in results
        assert "radius" in results
        assert "height" in results

        tf_mat = results["transform"]
        radius = results["radius"]
        height = results["height"]
        tf = jaxlie.SE3.from_matrix(tf_mat)

        cap = Cylinder.from_radius_and_height(
            radius=jnp.array([radius]),
            height=jnp.array([height]),
            transform=tf,
        )
        return cap

    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        cylinder = trimesh.creation.cylinder(
            radius=size[0].item(), height=size[1].item() * 2
        )
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        cylinder.vertices = trimesh.transform_points(cylinder.vertices, tf)
        return cylinder


@jdc.pytree_dataclass
class Convex(CollGeom):
    # Experimental. May be slightly buggy.
    mesh_info: jdc.Static[ConvexMesh]
    offset_to_origin: jdc.Static[Float[jax.Array, "*batch 3"]]

    @staticmethod
    def from_mesh(
        mesh: trimesh.Trimesh,
        n_verts: int = 32,
        batch_axes: tuple[int, ...] = (),
    ) -> Convex:
        """
        Create geometry from convex mesh.
        """
        # Make the meshes convex.
        mesh = mesh.convex_hull

        # TODO Check why we must ensure that the mesh includes the origin.
        # Found that this is necessary for convex collisions to work!
        offset_to_origin = mesh.centroid
        mesh.vertices -= mesh.centroid

        # Decimate the mesh, to make convex-convex feasible.
        mesh = Convex.decimate(mesh, n_verts=n_verts)
        mesh_info = Convex._get_mesh_mjx(mesh)

        tf = jaxlie.SE3.identity(batch_axes)

        # `size` isn't used for convex meshes -- and we can't use it for th mesh index either,
        # because that would mean we can have dynamically sized arrays (if N(mesh1) != N(mesh2)).
        size = jnp.zeros(batch_axes + (3,))

        return Convex(
            pose=tf,
            size=size,  # unused.
            mesh_info=mesh_info,
            offset_to_origin=offset_to_origin,
        )

    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        mesh = cast(
            trimesh.Trimesh,
            trimesh.PointCloud(
                self.mesh_info.vert,
            ).convex_hull,
        )
        mesh.fix_normals()
        mesh.vertices += self.offset_to_origin

        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        mesh.apply_transform(tf)
        return mesh

    @staticmethod
    def decimate(mesh: trimesh.Trimesh, n_verts: int) -> trimesh.Trimesh:
        """
        Decimate a mesh to have `n_vert` vertices, for:
        - stacking mesh information, and
        - faster collision checks.

        According to https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-the-sharp-bits,
        For reasonable performance, `n_vert` should be:
        - <200 for convex-primitive collisions.
        - <32 for convex-convex collisions.
        """
        hull = scipy.spatial.ConvexHull(mesh.vertices, qhull_options=f"TA{n_verts}")

        hull_verts = mesh.vertices[hull.vertices]
        hull_faces = onp.searchsorted(hull.vertices, hull.simplices.flatten()).reshape(
            -1, 3
        )

        _mesh = trimesh.Trimesh(vertices=hull_verts, faces=hull_faces)
        _mesh.fix_normals()
        return _mesh

    @staticmethod
    def _get_mesh_mjx(mesh) -> ConvexMesh:
        # Based on https://github.com/google-deepmind/mujoco/blob/43a7493d1739d5cb1618de6863f929d99c7a8822/mjx/mujoco/mjx/_src/mesh.py#L280.
        vert = onp.array(mesh.vertices)
        face = onp.array(mesh.faces)
        face_normal = _get_face_norm(vert, face)
        edge, edge_face_normal = _get_edge_normals(face, face_normal)
        face = vert[face]  # materialize full nface x nvert matrix

        return ConvexMesh(
            vert=jnp.array(vert),
            face=jnp.array(face),
            face_normal=jnp.array(face_normal),
            edge=jnp.array(edge),
            edge_face_normal=jnp.array(edge_face_normal),
        )
