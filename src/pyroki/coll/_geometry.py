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
import jax.scipy.ndimage

from ._utils import make_frame


@jdc.pytree_dataclass
class CollGeom(abc.ABC):
    """Base class for geometric objects."""

    pose: jaxlie.SE3
    size: Float[Array, "*batch shape_dim"]

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

    @staticmethod
    def from_trimesh(mesh: trimesh.Trimesh) -> Capsule:
        """
        Create Capsule geometry from minimum bounding cylinder of the mesh.
        """
        if mesh.is_empty:
            return Capsule(pose=jaxlie.SE3.identity(), size=jnp.zeros((2,)))
        results = trimesh.bounds.minimum_cylinder(mesh)
        radius = results["radius"]
        height = results["height"]
        tf_mat = results["transform"]
        tf = jaxlie.SE3.from_matrix(tf_mat)
        capsule = Capsule.from_center_radius_height(
            center=jnp.zeros((3,)),
            orientation_mat=jnp.eye(3),
            radius=radius,
            height=height,
        )
        capsule = capsule.transform(tf)
        return capsule

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

    def decompose_to_spheres(self, n_segments: int) -> Sphere:
        """
        Decompose the capsule into a series of spheres along its axis.
        Args: n_segments: Number of spheres.
        Returns: Sphere object shape (n_segments, *batch, ...).
        """
        batch_axes = self.get_batch_axes()
        radii = self.radius

        # Calculate local offsets for sphere centers along z-axis.
        segment_factors = jnp.linspace(-1.0, 1.0, n_segments)
        local_offsets_vec = jnp.array([0.0, 0.0, 1.0])[None, None, :] * (
            segment_factors[:, None, None] * self.length[None, ..., None]
        )

        # Create base spheres (at origin, correct radius) and transform them.
        spheres = Sphere.from_center_and_radius(
            center=jnp.zeros((n_segments,) + batch_axes + (3,)),
            radius=jnp.broadcast_to(radii, (n_segments,) + batch_axes),
        )

        # Broadcast capsule pose and apply transforms.
        capsule_pose_broadcast = jaxlie.SE3(
            jnp.broadcast_to(
                self.pose.wxyz_xyz,
                (n_segments,) + self.pose.get_batch_axes() + (7,),
            )
        )
        spheres = spheres.transform(
            capsule_pose_broadcast @ jaxlie.SE3.from_translation(local_offsets_vec)
        )
        assert spheres.get_batch_axes() == (n_segments,) + batch_axes
        return spheres

    @staticmethod
    def from_sphere_pairs(sph_0: Sphere, sph_1: Sphere) -> Capsule:
        """
        Create a capsule connecting the centers of two spheres.
        Args: sph_0, sph_1: Input spheres.
        Returns: Capsule object with the same batch shape.
        """
        assert sph_0.get_batch_axes() == sph_1.get_batch_axes(), "Batch axes mismatch"

        pos0 = sph_0.pose.translation()
        pos1 = sph_1.pose.translation()
        vec = pos1 - pos0

        length_sq = jnp.sum(vec**2, axis=-1, keepdims=True)
        height = jnp.sqrt(length_sq).squeeze(-1)

        transform = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3.from_matrix(make_frame(vec)),
            translation=(pos0 + pos1) / 2.0,
        )

        capsule = Capsule.from_center_radius_height(
            center=transform.translation(),
            orientation_mat=transform.rotation().as_matrix(),
            radius=sph_0.radius,
            height=height,
        )

        assert capsule.get_batch_axes() == sph_0.get_batch_axes()
        return capsule


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


@jdc.pytree_dataclass
class Heightmap(CollGeom):
    """Heightmap geometry defined by a grid of height values.
    The heightmap is oriented such that its base lies on the XY plane of its local frame.

    size[*batch, 0] = x_scale (grid spacing along local x)
    size[*batch, 1] = y_scale (grid spacing along local y)
    size[*batch, 2] = height_scale (multiplier for height data)
    """

    height_data: Float[Array, "*batch H W"]

    @property
    def x_scale(self) -> Float[Array, "*batch"]:
        """Grid spacing along the local X-axis."""
        return self.size[..., 0]

    @property
    def y_scale(self) -> Float[Array, "*batch"]:
        """Grid spacing along the local Y-axis."""
        return self.size[..., 1]

    @property
    def height_scale(self) -> Float[Array, "*batch"]:
        """Multiplier applied to height data values."""
        return self.size[..., 2]

    @property
    def rows(self) -> int:
        """Number of rows in the height grid (along local Y)."""
        return self.height_data.shape[-2]

    @property
    def cols(self) -> int:
        """Number of columns in the height grid (along local X)."""
        return self.height_data.shape[-1]

    def _interpolate_height_at_coords(
        self,
        world_coords: Float[Array, "*batch 3"],
    ) -> Float[Array, "*batch"]:
        """Interpolates heightmap surface height at given world coordinates.

        Args:
            world_coords: Coordinates in the world frame (*batch, 3).

        Returns:
            Interpolated heightmap surface height in the heightmap's local frame (*batch).
        """
        # Transform world coords to heightmap local frame.
        local_coords = self.pose.inverse().apply(world_coords)
        sx, sy = local_coords[..., 0], local_coords[..., 1]

        # Calculate continuous grid indices (r, c) from local coords (sx, sy)
        # Origin (0,0) in local frame corresponds to center of the base grid!
        c_cont = sx / self.x_scale + (self.cols - 1) / 2.0
        r_cont = sy / self.y_scale + (self.rows - 1) / 2.0

        # Interpolate height data at (r_cont, c_cont).
        batch_axes = self.get_batch_axes()
        # Ensure batch axes of coords match heightmap's batch axes.
        target_batch_shape = jnp.broadcast_shapes(batch_axes, world_coords.shape[:-1])
        coords_bc = jnp.broadcast_to(
            jnp.stack([r_cont, c_cont], axis=-1), target_batch_shape + (2,)
        )
        hm_data_bc = jnp.broadcast_to(
            self.height_data, target_batch_shape + self.height_data.shape[-2:]
        )

        if target_batch_shape:
            batch_size = onp.prod(target_batch_shape).item()
            # Reshape for vmap.
            h_data_flat = hm_data_bc.reshape((batch_size, self.rows, self.cols))
            coords_flat = coords_bc.reshape((batch_size, 2))

            # vmap over flattened batch dimension.
            vmap_interpolate = jax.vmap(
                lambda h, c: jax.scipy.ndimage.map_coordinates(
                    h, c[:, None], order=1, mode='nearest'
                ).squeeze(),
                in_axes=(0, 0)
            )
            interpolated_heights_flat = vmap_interpolate(h_data_flat, coords_flat)
            interpolated_heights = interpolated_heights_flat.reshape(target_batch_shape)
        else:
            # Non-batched case.
            interpolated_heights = jax.scipy.ndimage.map_coordinates(
                hm_data_bc,
                (coords_bc[0:1], coords_bc[1:2]), # ([r_cont], [c_cont])
                order=1, mode='nearest'
            ).squeeze()

        # Scale interpolated height
        interpolated_local_z = interpolated_heights * self.height_scale
        return interpolated_local_z

    def _get_vertices_local(self) -> Float[Array, "*batch H*W 3"]:
        """Computes the heightmap vertices in its local frame using JAX.

        Returns:
            Vertices array with shape (*batch, H*W, 3).
        """
        batch_axes = self.get_batch_axes()
        H, W = self.rows, self.cols

        # Create grid coordinates (centered).
        x = (jnp.arange(W) - (W - 1) / 2.0) * self.x_scale[..., None]
        y = (jnp.arange(H) - (H - 1) / 2.0) * self.y_scale[..., None]

        # Add batch dimensions for meshgrid if necessary.
        if batch_axes:
            x = jnp.broadcast_to(x, batch_axes + (W,))
            y = jnp.broadcast_to(y, batch_axes + (H,))
            xx, yy = jnp.meshgrid(x, y, indexing='xy') # Results shape (*batch, H, W).
        else:
            xx, yy = jnp.meshgrid(x, y, indexing='xy') # Results shape (H, W).

        # Scale height data.
        zz = self.height_data * self.height_scale[..., None, None]

        # Combine into vertices: (*batch, H, W, 3).
        vertices = jnp.stack([xx, yy, zz], axis=-1)

        # Reshape to (*batch, H*W, 3).
        vertices_flat = vertices.reshape(batch_axes + (H * W, 3))
        return vertices_flat

    def broadcast_to(self, *shape: int) -> Self:
        """Broadcast geometry to given shape."""
        new_pose_wxyz_xyz = jnp.broadcast_to(self.pose.wxyz_xyz, shape + (7,))
        new_pose = jaxlie.SE3(new_pose_wxyz_xyz)
        shape_dim = self.size.shape[-1]
        new_size = jnp.broadcast_to(self.size, shape + (shape_dim,))
        new_height_data = jnp.broadcast_to(
            self.height_data, shape + self.height_data.shape[-2:]
        )
        return type(self)(pose=new_pose, size=new_size, height_data=new_height_data)

    def reshape(self, *shape: int) -> Self:
        """Reshape geometry to given shape."""
        new_pose_wxyz_xyz = self.pose.wxyz_xyz.reshape(shape + (7,))
        new_pose = jaxlie.SE3(new_pose_wxyz_xyz)
        shape_dim = self.size.shape[-1]
        new_size = self.size.reshape(shape + (shape_dim,))
        new_height_data = self.height_data.reshape(
            shape + self.height_data.shape[-2:]
        )
        return type(self)(pose=new_pose, size=new_size, height_data=new_height_data)

    def transform(self, transform: jaxlie.SE3) -> Self:
        """Applies an SE3 transformation to the geometry."""
        new_pose = transform @ self.pose
        new_batch_axes = new_pose.get_batch_axes()
        broadcast_size = jnp.broadcast_to(
            self.size, new_batch_axes + self.size.shape[-1:]
        )
        broadcast_height_data = jnp.broadcast_to(
            self.height_data, new_batch_axes + self.height_data.shape[-2:]
        )
        return type(self)(
            pose=new_pose,
            size=broadcast_size,
            height_data=broadcast_height_data,
        )

    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        """Create a single trimesh object from height data at a given index.
        Also includes back-facing triangles for two-sided rendering.
        """
        pose_i: jaxlie.SE3 = jax.tree_map(lambda x: x[index], self.pose)
        height_data_i: Float[Array, "H W"] = self.height_data[index]
        x_scale_i = float(self.x_scale[index])
        y_scale_i = float(self.y_scale[index])
        height_scale_i = float(self.height_scale[index])

        rows, cols = height_data_i.shape
        if rows < 2 or cols < 2:
            # Need at least a 2x2 grid to form a face.
            return trimesh.Trimesh()

        # Create vertex grid.
        x = onp.arange(cols) * x_scale_i
        y = onp.arange(rows) * y_scale_i
        xx, yy = onp.meshgrid(x, y)
        zz = onp.array(height_data_i) * height_scale_i

        vertices = onp.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        # Center the vertices around the origin before applying pose.
        center_offset = onp.array(
            [(cols - 1) * x_scale_i / 2.0, (rows - 1) * y_scale_i / 2.0, 0.0]
        )
        vertices -= center_offset

        # Create faces (triangles) - both front and back.
        front_faces = []
        back_faces = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                idx0 = r * cols + c
                idx1 = r * cols + (c + 1)
                idx2 = (r + 1) * cols + c
                idx3 = (r + 1) * cols + (c + 1)
                front_faces.append([idx0, idx1, idx2])  # Triangle 1 (front)
                front_faces.append([idx1, idx3, idx2])  # Triangle 2 (front)
                back_faces.append([idx0, idx2, idx1])   # Triangle 1 (back)
                back_faces.append([idx1, idx2, idx3])   # Triangle 2 (back)

        all_faces = front_faces + back_faces

        if not all_faces:
            return trimesh.Trimesh()

        heightmap_mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces)

        tf = onp.array(pose_i.as_matrix())
        heightmap_mesh.apply_transform(tf)

        return heightmap_mesh

