"""
Differentiable robot collision model, implemented in JAX.
"""

from __future__ import annotations
from typing import Callable, Optional, Sequence, cast

from loguru import logger

import trimesh
import trimesh.bounds
import yourdfpy

import jax
from jax import Array
import jax.numpy as jnp
import jaxlie

from jaxtyping import Float, Int
import jax_dataclasses as jdc

from jaxmp.kinematics import JaxKinTree
from jaxmp.coll._collide_types import Capsule, CollGeom, Convex
from jaxmp.coll._collide import collide


def _capsules_from_meshes(meshes: Sequence[trimesh.Trimesh]) -> Capsule:
    capsules = [Capsule.from_min_cylinder(mesh) for mesh in meshes]
    return jax.tree.map(lambda *args: jnp.stack(args), *capsules)


@jdc.pytree_dataclass
class RobotColl:
    """Collision model for a robot, which can be put into different configurations.
    For optimization, we assume that `coll` is a single `CollGeom`.
    """

    num_colls: jdc.Static[int]

    coll: CollGeom | Sequence[CollGeom]
    """Collision model for the robot, either a single `CollGeom` or a list of them."""

    coll_link_names: jdc.Static[tuple[str]]
    """Names of the links in the robot, length `links`."""

    link_joint_idx: jdc.Static[Int[Array, " colls"]]
    """Index of the parent joint for each collision body."""

    link_to_colls: jdc.Static[dict[int, jax.Array]]
    """Mapping from each link to the indices of its collision bodies."""

    self_coll_list: jdc.Static[Sequence[tuple[int, int]]]
    """Collision matrix, where we store the list of colliding links."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        coll_handler: Callable[
            [Sequence[trimesh.Trimesh]], CollGeom | Sequence[CollGeom]
        ] = _capsules_from_meshes,
        self_coll_ignore: Optional[list[tuple[str, str]]] = None,
        ignore_immediate_parent: bool = True,
    ):
        """
        Build a differentiable robot collision model from a URDF.

        Args:
            urdf: The URDF object.
            self_coll_ignore: List of tuples of link names that are allowed to collide.
            ignore_immediate_parent: If True, ignore collisions between parent and child links.
        """

        # Re-load urdf, but with the collision data.
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access
        urdf = yourdfpy.URDF(
            robot=urdf.robot,
            filename_handler=filename_handler,
            load_collision_meshes=True,
        )

        # Gather all the collision links.
        coll_link_meshes = list[trimesh.Trimesh]()
        link_joint_idx = list[int]()
        link_names = list[str]()
        link_to_colls = dict[int, jax.Array]()

        if self_coll_ignore is None:
            self_coll_ignore = []

        # Get all collision links.
        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            curr_link = joint.child
            assert curr_link in urdf.link_map

            coll_link = RobotColl._get_coll_links(urdf, curr_link)
            if len(coll_link) == 0:
                continue

            # Add the collision links to the list.
            # 1. First, note the index of the current link.
            link_names.append(curr_link)
            coll_idx = len(link_names) - 1  # Last link.

            # 2. Add the collision links to the list.
            coll_link_meshes.extend(coll_link)
            link_joint_idx.extend([joint_idx] * len(coll_link))

            # 3. Update the mapping from link to collision bodies.
            end = len(link_joint_idx)
            start = end - len(coll_link)
            link_to_colls[coll_idx] = jnp.arange(start, end)

            if ignore_immediate_parent:
                self_coll_ignore.append((joint.parent, joint.child))

        assert len(coll_link_meshes) > 0, "No collision links found in URDF."
        logger.info("Found {} collision bodies", len(coll_link_meshes))

        coll_links = coll_handler(coll_link_meshes)
        if isinstance(coll_links, CollGeom):
            assert len(coll_links.get_batch_axes()) == 1

        num_colls = len(link_joint_idx)
        link_joint_idx = jnp.array(link_joint_idx)
        link_names = tuple[str](link_names)

        self_coll_list = RobotColl.create_self_coll_list(
            link_names,
            self_coll_ignore,
        )

        return RobotColl(
            num_colls=num_colls,
            coll=coll_links,
            coll_link_names=link_names,
            link_joint_idx=link_joint_idx,
            link_to_colls=link_to_colls,
            self_coll_list=self_coll_list,
        )

    @staticmethod
    def _get_coll_links(
        urdf: yourdfpy.URDF, curr_link: str
    ) -> Sequence[trimesh.Trimesh]:
        """
        Get the `CapsuleColl` collision primitives for a given link.
        """
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access

        coll_mesh_list = urdf.link_map[curr_link].collisions
        if len(coll_mesh_list) == 0:
            return []

        coll_link_mesh = []
        for coll in coll_mesh_list:
            # Handle different geometry types.
            coll_mesh: Optional[trimesh.Trimesh] = None
            geom = coll.geometry
            if geom.box is not None:
                coll_mesh = trimesh.creation.box(extents=geom.box.size)
            elif geom.cylinder is not None:
                coll_mesh = trimesh.creation.cylinder(
                    radius=geom.cylinder.radius, height=geom.cylinder.length
                )
            elif geom.sphere is not None:
                coll_mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
            elif geom.mesh is not None:
                coll_mesh = cast(
                    trimesh.Trimesh,
                    trimesh.load(
                        file_obj=filename_handler(geom.mesh.filename), force="mesh"
                    ),
                )
                coll_mesh.fix_normals()

            if coll_mesh is None:
                raise ValueError(f"No collision mesh found for {curr_link}!")
            coll_link_mesh.append(coll_mesh)

        return coll_link_mesh

    def make_self_coll_params(
        self,
        default: float,
        override_weights: dict[tuple[str, str], float] = {},
    ) -> jax.Array:
        values = jnp.full((len(self.self_coll_list),), default)
        for (name_0, name_1), weight in override_weights.items():
            idx_0 = self.coll_link_names.index(name_0)
            idx_1 = self.coll_link_names.index(name_1)
            if (idx_0, idx_1) in self.self_coll_list:
                values = values.at[self.self_coll_list.index((idx_0, idx_1))].set(
                    weight
                )
            if (idx_1, idx_0) in self.self_coll_list:
                values = values.at[self.self_coll_list.index((idx_1, idx_0))].set(
                    weight
                )
        return values

    def make_world_coll_params(
        self,
        default: float,
        override_weights: dict[str, float] = {},
    ) -> jax.Array:
        values = jnp.full((len(self.coll_link_names),), default)
        for name, weight in override_weights.items():
            idx = self.coll_link_names.index(name)
            values = values.at[idx].set(weight)
        return values

    @staticmethod
    def create_self_coll_list(
        coll_link_names: tuple[str],
        self_coll_ignore: list[tuple[str, str]],
    ) -> Sequence[tuple[int, int]]:
        """
        Create a collision matrix for the robot, where `coll_matrix[i, j] == 1`.
        """

        def check_coll(i: int, j: int) -> bool:
            """Remove self- and adjacent link collisions."""
            if i == j:
                return False
            if (coll_link_names[i], coll_link_names[j]) in self_coll_ignore:
                return False
            if (coll_link_names[j], coll_link_names[i]) in self_coll_ignore:
                return False

            return True

        coll_list = []
        for i in range(len(coll_link_names)):
            for j in range(len(coll_link_names)):
                if i <= j:
                    continue
                if check_coll(i, j):
                    coll_list.append((i, j))
        return coll_list

    def at_joints(
        self, kin: JaxKinTree, cfg: Float[jax.Array, "*batch joints"]
    ) -> Float[CollGeom, "*batch links"] | Sequence[CollGeom]:
        """Get the collision model for the robot at a given configuration."""
        Ts_joint_world = kin.forward_kinematics(cfg)[..., self.link_joint_idx, :]

        if isinstance(self.coll, CollGeom):
            coll = self.coll.transform(jaxlie.SE3(Ts_joint_world))
        else:
            coll = [
                coll.transform(jaxlie.SE3(Ts_joint_world[..., idx, :]))
                for idx, coll in enumerate(self.coll)
            ]

        return coll

    def self_coll_dist(
        self, kin: JaxKinTree, cfg: Float[jax.Array, "*batch joints"]
    ) -> Float[jax.Array, "*batch"]:
        """Get the minimum distance between the robot's collision bodies."""
        batch_size = cfg.shape[:-1]
        coll = self.at_joints(kin, cfg)
        if isinstance(coll, CollGeom):
            dist = collide(
                coll.reshape(*batch_size, 1, -1), coll.reshape(*batch_size, -1, 1)
            ).dist
            coll_list = jnp.array(self.self_coll_list)
            dist = dist[..., coll_list[:, 0], coll_list[:, 1]]
            dist = dist.min(axis=-1)
            assert dist.shape == batch_size
            return dist
        else:
            if isinstance(coll[0], Convex):
                logger.warning(
                    "Convex collisions are less reliable, consider using capsules."
                )
            min_dist = jnp.full((*batch_size, len(self.self_coll_list)), jnp.inf)
            for idx, (link_0, link_1) in enumerate(self.self_coll_list):
                for coll_0 in self.link_to_colls[link_0]:
                    for coll_1 in self.link_to_colls[link_1]:
                        dist = collide(coll[coll_0], coll[coll_1]).dist
                        min_dist = min_dist.at[..., idx].set(
                            jnp.minimum(dist, min_dist[..., idx])
                        )
                        if dist.min() < 0:
                            print(
                                self.coll_link_names[link_0],
                                self.coll_link_names[link_1],
                            )
            assert not jnp.any(jnp.isinf(min_dist))
            min_dist = min_dist.min(axis=-1)
            assert min_dist.shape == batch_size
            return min_dist

    def world_coll_dist(
        self, kin: JaxKinTree, cfg: Float[jax.Array, "*batch joints"], world: CollGeom
    ) -> Float[jax.Array, "*batch"]:
        batch_size = cfg.shape[:-1]
        coll = self.at_joints(kin, cfg)
        if isinstance(coll, CollGeom):
            dist = collide(coll, world).dist
            dist = dist.min(axis=-1)
            assert dist.shape == batch_size
            return dist
        else:
            if isinstance(coll[0], Convex):
                logger.warning(
                    "Convex collisions are less reliable, consider using capsules."
                )
            min_dist = jnp.full((*batch_size, len(coll)), jnp.inf)
            for idx, c in enumerate(coll):
                dist = collide(c, world).dist
                min_dist = min_dist.at[..., idx].set(
                    jnp.minimum(dist, min_dist[..., idx])
                )
            assert not jnp.any(jnp.isinf(min_dist))
            min_dist = min_dist.min(axis=-1)
            assert min_dist.shape == batch_size
            return min_dist
