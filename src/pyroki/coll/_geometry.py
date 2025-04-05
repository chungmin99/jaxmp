from __future__ import annotations

import abc
from typing import cast, Self

import trimesh

import jax.numpy as jnp
import jaxlie
from jaxtyping import Float, Array
import jax_dataclasses as jdc
import numpy as onp
import jax

from ._utils import make_frame


@jdc.pytree_dataclass
class CollGeom(abc.ABC):
    """Base class for geometric objects."""

    pose: jaxlie.SE3
    size: Float[Array, "*batch shape_dim"]

    def __post_init__(self):
        """Check for compatible batch dimensions."""
        self.get_batch_axes()  # Will raise ValueError if incompatible

    def get_batch_axes(self) -> tuple[int, ...]:
        """Get batch axes of the geometry."""
        batch_axes_from_pose = self.pose.get_batch_axes()
        size_batch_axes = self.size.shape[:-1]
        assert (
            size_batch_axes == batch_axes_from_pose
        ), f"Size batch axes {size_batch_axes} do not match pose batch axes {batch_axes_from_pose}."
        return batch_axes_from_pose

    def broadcast_to(self, *shape: int) -> Self:
        """Broadcast geometry to given shape."""
        new_pose_wxyz_xyz = jnp.broadcast_to(self.pose.wxyz_xyz, shape + (7,))
        new_pose = jaxlie.SE3(new_pose_wxyz_xyz)
        shape_dim = self.size.shape[-1]
        new_size = jnp.broadcast_to(self.size, shape + (shape_dim,))
        return type(self)(pose=new_pose, size=new_size)

    def reshape(self, *shape: int) -> Self:
        """Reshape geometry to given shape."""
        new_pose_wxyz_xyz = self.pose.wxyz_xyz.reshape(shape + (7,))
        new_pose = jaxlie.SE3(new_pose_wxyz_xyz)
        shape_dim = self.size.shape[-1]
        new_size = self.size.reshape(shape + (shape_dim,))
        return type(self)(pose=new_pose, size=new_size)

    def transform(self, transform: jaxlie.SE3) -> Self:
        """Applies an SE3 transformation to the geometry."""
        new_pose = transform @ self.pose
        new_batch_axes = new_pose.get_batch_axes()
        broadcast_size = jnp.broadcast_to(
            self.size, new_batch_axes + self.size.shape[-1:]
        )
        kwargs = {"pose": new_pose, "size": broadcast_size}
        return type(self)(**kwargs)

    @abc.abstractmethod
    def _create_one_mesh(self, index: tuple[int, ...]) -> trimesh.Trimesh:
        """Helper to create a single trimesh object from batch data at a given index."""
        raise NotImplementedError

    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert the (potentially batched) geometry to a single trimesh object."""
        batch_axes = self.get_batch_axes()
        if not batch_axes:
            return self._create_one_mesh(tuple())

        meshes = [
            self._create_one_mesh(idx_tuple) for idx_tuple in onp.ndindex(batch_axes)
        ]
        if not meshes:
            return trimesh.Trimesh()

        return cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))


@jdc.pytree_dataclass
class HalfSpace(CollGeom):
    """HalfSpace geometry defined by a point and an outward normal. Size is ignored."""

    @property
    def normal(self) -> Float[Array, "*batch 3"]:
        """Normal vector (Z-axis of rotation matrix)."""
        return self.pose.rotation().as_matrix()[..., :, 2]

    @property
    def offset(self) -> Float[Array, "*batch"]:
        """Offset from origin along the normal (origin = point on plane)."""
        # For a plane defined by p_0 and n, the offset is dot(p_0, n)
        # Here, pose.translation() is p_0
        return jnp.einsum("...i,...i->...", self.normal, self.pose.translation())

    @staticmethod
    def from_point_and_normal(
        point: Float[Array, "*batch 3"], normal: Float[Array, "*batch 3"]
    ) -> HalfSpace:
        """Create a HalfSpace geometry from a point on the boundary and outward normal."""
        batch_axes = jnp.broadcast_shapes(point.shape[:-1], normal.shape[:-1])
        point = jnp.broadcast_to(point, batch_axes + (3,))
        normal = jnp.broadcast_to(normal, batch_axes + (3,))
        mat = make_frame(normal)
        pos = point
        pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.from_matrix(mat), pos)
        size = jnp.zeros(batch_axes + (1,), dtype=pos.dtype)
        return HalfSpace(pose=pose, size=size)

    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        """Visualize HalfSpace as a large thin box aligned with its boundary plane."""
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        mat = onp.array(pose_i.rotation().as_matrix())
        # Visualize as a box representing the boundary plane
        plane_mesh = trimesh.creation.box(extents=[10, 10, 0.01])
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        plane_mesh.apply_transform(tf)
        return plane_mesh


@jdc.pytree_dataclass
class Sphere(CollGeom):
    """Sphere geometry. size[*batch, 0] = radius."""

    @property
    def radius(self) -> Float[Array, "*batch"]:
        """Radius of the sphere."""
        return self.size[..., 0]

    @staticmethod
    def from_center_and_radius(
        center: Float[Array, "*batch 3"], radius: Float[Array, "*batch"]
    ) -> Sphere:
        """Create a Sphere geometry from a center point and radius."""
        batch_axes = jnp.broadcast_shapes(center.shape[:-1], radius.shape)
        center = jnp.broadcast_to(center, batch_axes + (3,))
        radius = jnp.broadcast_to(radius, batch_axes)
        pos = center
        # Create identity pose for sphere
        num_batch_elements = onp.prod(batch_axes).item() if batch_axes else 1
        quat_wxyz = jnp.stack(
            [jnp.array([1.0, 0.0, 0.0, 0.0], dtype=pos.dtype)] * num_batch_elements,
            axis=0,
        )
        quat_wxyz = quat_wxyz.reshape(batch_axes + (4,))
        wxyz_xyz = jnp.concatenate([quat_wxyz, pos], axis=-1)
        pose = jaxlie.SE3(wxyz_xyz)

        # Store radius in size[..., 0], shape_dim=1
        size = radius[..., None]
        return Sphere(pose=pose, size=size)

    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        pose_i: jaxlie.SE3 = jax.tree_map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        radius_val = float(self.radius[index])
        sphere_mesh = trimesh.creation.icosphere(radius=radius_val)
        # Only apply translation for sphere
        tf = onp.eye(4)
        tf[:3, 3] = pos
        sphere_mesh.apply_transform(tf)
        return sphere_mesh


@jdc.pytree_dataclass
class Capsule(CollGeom):
    """Capsule geometry. size[*batch, 0]=radius, size[*batch, 1]=half-length."""

    @property
    def radius(self) -> Float[Array, "*batch"]:
        """Radius of the capsule ends and cylinder."""
        return self.size[..., 0]

    @property
    def length(self) -> Float[Array, "*batch"]:
        """Half-length of the cylindrical segment."""
        return self.size[..., 1]

    @property
    def axis(self) -> Float[Array, "*batch 3"]:
        """Axis direction (Z-axis of rotation matrix)."""
        return self.pose.rotation().as_matrix()[..., :, 2]

    @staticmethod
    def from_center_radius_height(
        center: Float[Array, "*batch 3"],
        orientation_mat: Float[Array, "*batch 3 3"],
        radius: Float[Array, "*batch"],
        height: Float[Array, "*batch"],  # Full height
    ) -> Capsule:
        """Create Capsule geometry from center, orientation, radius, and *full* height."""
        batch_axes = jnp.broadcast_shapes(
            center.shape[:-1], orientation_mat.shape[:-2], radius.shape, height.shape
        )
        pos = jnp.broadcast_to(center, batch_axes + (3,))
        mat = jnp.broadcast_to(orientation_mat, batch_axes + (3, 3))
        radius = jnp.broadcast_to(radius, batch_axes)
        height = jnp.broadcast_to(height, batch_axes)

        pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.from_matrix(mat), pos)

        # Store radius and half-length, shape_dim=2
        size = jnp.stack([radius, height / 2.0], axis=-1)
        return Capsule(pose=pose, size=size)

    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        pose_i: jaxlie.SE3 = jax.tree_map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        mat = onp.array(pose_i.rotation().as_matrix())
        radius_val = float(self.radius[index])
        height_val = float(self.length[index]) * 2  # Trimesh expects full height
        capsule_mesh = trimesh.creation.capsule(radius=radius_val, height=height_val)
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        capsule_mesh.apply_transform(tf)
        return capsule_mesh


@jdc.pytree_dataclass
class Box(CollGeom):
    """Box geometry. size[*batch, 3] = full extents (lx, ly, lz) along local axes."""

    @property
    def extents(self) -> Float[Array, "*batch 3"]:
        """Full extents (size) of the box along its local axes."""
        return self.size

    @staticmethod
    def from_center_extents_pose(
        pose: jaxlie.SE3,
        extents: Float[Array, "*batch 3"],  # lx, ly, lz
    ) -> Box:
        """Create Box geometry from center pose and full extents."""
        batch_axes = pose.get_batch_axes()
        # Ensure extents are broadcastable to batch_axes + (3,)
        try:
            broadcast_extents_shape = jnp.broadcast_shapes(
                batch_axes + (3,), extents.shape
            )
            assert broadcast_extents_shape[:-1] == batch_axes
        except ValueError:
            raise ValueError(
                f"Extents shape {extents.shape} incompatible with pose batch axes {batch_axes}"
            )
        # Broadcast size to match pose batch shape
        size = jnp.broadcast_to(extents, batch_axes + (3,))
        return Box(pose=pose, size=size)

    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        """Create a trimesh box for one element in the batch."""
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        extents_i = onp.array(self.extents[index])  # Get extents for this index
        pos = onp.array(pose_i.translation())
        mat = onp.array(pose_i.rotation().as_matrix())

        # Create box centered at origin with given extents
        box_mesh = trimesh.creation.box(extents=extents_i)

        # Apply transform
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        box_mesh.apply_transform(tf)
        return box_mesh
