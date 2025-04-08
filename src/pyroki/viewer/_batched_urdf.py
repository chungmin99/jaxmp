import yourdfpy
import viser
import viser.transforms as vtf

import numpy as onp
import jax.numpy as jnp
import jaxlie

from pyroki import Robot


class BatchedURDF:
    """
    Helper for rendering batched URDFs in Viser.
    Similar to `viser.extras.ViserUrdf`, but batched using `pyroki`'s batched forward kinematics.

    Args:
        target: Viser server or client handle to add URDF to.
        urdf: URDF to render.
        root_node_name: Name of the root node in the Viser scene.
    """

    def __init__(
        self,
        target: viser.ViserServer | viser.ClientHandle,
        urdf: yourdfpy.URDF,
        root_node_name: str = "/",
    ):
        assert root_node_name.startswith("/")
        robot = Robot.from_urdf(urdf)

        self._urdf = urdf
        self._robot = robot
        self._target = target
        self._root_node_name = root_node_name

        self._populate()

    def _populate(self):
        dummy_transform = vtf.SE3.identity(batch_axes=(1,))
        dummy_position = dummy_transform.translation()
        dummy_wxyz = dummy_transform.rotation().wxyz

        self._meshes: dict[str, list[viser.BatchedGlbHandle]] = {}
        self._link_to_meshes: dict[str, onp.ndarray] = {}

        for mesh_name, mesh in self._urdf.scene.geometry.items():
            link_name = self._urdf.scene.graph.transforms.parents[mesh_name]
            if link_name not in self._meshes:
                self._meshes[link_name] = []

            # Put mesh in the link frame.
            T_parent_child = self._urdf.get_transform(
                mesh_name, self._urdf.scene.graph.transforms.parents[mesh_name]
            )
            mesh = mesh.copy()
            mesh.apply_transform(T_parent_child)

            self._meshes[link_name].append(
                self._target.scene.add_batched_meshes_trimesh(
                    f"{self._root_node_name}/{mesh_name}",
                    mesh,
                    batched_positions=dummy_position,
                    batched_wxyzs=dummy_wxyz,
                )
            )

            self._link_to_meshes[link_name] = T_parent_child

    def remove(self):
        for meshes in self._meshes.values():
            for mesh in meshes:
                mesh.remove()

    def update_cfg(self, cfg: jnp.ndarray):
        Ts_link_world = self._robot.forward_kinematics_links(cfg)
        for link_name, meshes in self._meshes.items():
            link_idx = self._robot.link.names.index(link_name)
            T_mesh_world = jaxlie.SE3(Ts_link_world[link_idx])

            position = onp.array(T_mesh_world.translation())
            wxyz = onp.array(T_mesh_world.rotation().wxyz)
            for mesh in meshes:
                mesh.batched_positions = position
                mesh.batched_wxyzs = wxyz
