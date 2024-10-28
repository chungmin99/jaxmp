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
from jaxmp.coll._collide_types import Capsule, CollGeom


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

    link_joint_idx: jdc.Static[Int[Array, "colls"]]
    """Index of the parent joint for each collision body."""

    link_coll_idx: jdc.Static[Int[Array, "colls"]]
    """Index of the link for each collision body, in `coll_link_names`."""

    self_coll_list: jdc.Static[Sequence[tuple[int, int]]]
    """Collision matrix, where we store the list of colliding bodies."""

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
        link_coll_idx = list[int]()

        if self_coll_ignore is None:
            self_coll_ignore = []

        # Get all collision links.
        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            curr_link = joint.child
            assert curr_link in urdf.link_map

            coll_link = RobotColl._get_coll_links(urdf, curr_link)
            if coll_link is None:
                continue

            coll_idx = len(link_names)
            link_names.append(curr_link)

            coll_link_meshes.extend(coll_link)
            link_joint_idx.extend([joint_idx] * len(coll_link))
            link_coll_idx.extend([coll_idx] * len(coll_link))

            if ignore_immediate_parent:
                self_coll_ignore.append((joint.parent, joint.child))

        assert len(coll_link_meshes) > 0, "No collision links found in URDF."
        logger.info("Found {} collision bodies", len(coll_link_meshes))

        coll_links = coll_handler(coll_link_meshes)

        num_colls = len(link_joint_idx)
        link_coll_idx = jnp.array(link_coll_idx)
        link_joint_idx = jnp.array(link_joint_idx)
        link_names = tuple[str](link_names)

        self_coll_list = RobotColl.create_self_coll_list(
            link_names,
            self_coll_ignore,
            link_coll_idx,
        )

        return RobotColl(
            num_colls=num_colls,
            coll=coll_links,
            coll_link_names=link_names,
            link_joint_idx=link_joint_idx,
            link_coll_idx=link_coll_idx,
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

    def coll_weight(
        self,
        weight: float = 1.0,
        override_weights: dict[str, float] = {},
    ) -> Float[Array, "colls"]:
        """Get the collision weight for each sphere."""
        coll_weights = jnp.full((self.num_colls,), weight)
        for name, weight in override_weights.items():
            idx = self.coll_link_names.index(name)
            coll_weights = jnp.where(
                self.link_coll_idx == idx, jnp.array(weight), coll_weights
            )
        return jnp.array(coll_weights)

    @staticmethod
    def create_self_coll_list(
        coll_link_names: tuple[str],
        self_coll_ignore: list[tuple[str, str]],
        link_coll_idx: Int[Array, "colls"],
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
        for i, idx_0 in enumerate(link_coll_idx):
            for j, idx_1 in enumerate(link_coll_idx):
                if i <= j:
                    continue
                if check_coll(idx_0, idx_1):
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
