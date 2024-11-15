"""
Data structure for multi-part objects with trajectory data, from RSRD.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
import jax
import viser
import trimesh

from loguru import logger

from jax import Array
import jax.numpy as jnp
import numpy as onp
import jax_dataclasses as jdc
from jaxtyping import Float
import jaxlie

from jaxmp.extras import AntipodalGrasps


@jdc.pytree_dataclass
class RSRDObject:
    means: Float[Array, "N 3"]
    quats: Float[Array, "N 4"]
    _scales: Float[Array, "N 3"]
    _colors: Float[Array, "N 3"]
    _opacities: Float[Array, "N 1"]

    num_groups: jdc.Static[int]
    timesteps: jdc.Static[int]

    group_labels: Float[Array, " N"]
    init_p2o: Float[Array, "N_groups 7"]
    part_deltas: Float[Array, "T N_groups 7"]
    _ns_to_world_scale: jdc.Static[float]

    single_hand_assignments: jdc.Static[tuple[int] | None] = None
    bimanual_assignments: jdc.Static[tuple[tuple[int, int]] | None] = None
    grasps: AntipodalGrasps | None = None

    @staticmethod
    def from_data(
        data_path: Path,
        prng_key: jax.Array,
        *,
        make_assignments: bool = True,
        make_grasps: bool = True,
        max_grasps_per_part: int = 20,
        min_width: float = 0.005,
        max_width: float = 0.05,
        max_depth: float = 0.08,
        nms_angle: float = jnp.pi / 8,
        nms_distance: float = 0.005,
    ) -> RSRDObject:
        data = json.loads(data_path.read_text())
        means = jnp.array(data["means"])
        quats = jnp.array(data["quats"])
        scales = jnp.array(data["scales"])
        colors = jnp.array(data["colors"])
        opacities = jnp.array(data["opacities"])
        group_labels = jnp.array(data["group_labels"])
        init_p2o = jnp.array(data["init_p2o"])
        part_deltas = jnp.array(data["part_deltas"])
        ns_to_world_scale = data["ns_to_world_scale"]

        num_groups = int(group_labels.max() + 1)
        timesteps = part_deltas.shape[0]

        # Reshape based on ns_to_world_scale.
        means *= ns_to_world_scale
        init_p2o = init_p2o.at[:, 4:].mul(ns_to_world_scale)
        part_deltas = part_deltas.at[:, :, 4:].mul(ns_to_world_scale)

        _self = RSRDObject(
            means,
            quats,
            scales,
            colors,
            opacities,
            num_groups,
            timesteps,
            group_labels,
            init_p2o,
            part_deltas,
            ns_to_world_scale,
        )

        if make_assignments:
            start = time.time()
            single_hand_assignments = _self.rank_parts_to_move_single(data)
            bimanual_assignments = _self.rank_parts_to_move_bimanual(data)
            logger.info(f"Ranking took {time.time() - start:.2f} seconds.")
        else:
            single_hand_assignments = None
            bimanual_assignments = None

        if make_grasps:
            start = time.time()
            grasp_list = []
            for group_idx in range(_self.num_groups):
                part = _self.get_part(group_idx)
                convex = trimesh.PointCloud(part.means).convex_hull
                grasps = AntipodalGrasps.from_sample_mesh(
                    convex,
                    prng_key=prng_key,
                    min_width=min_width,
                    max_width=max_width,
                    max_depth=max_depth,
                )
                grasps = grasps.nms(nms_distance, nms_angle)
                grasps = jax.tree.map(
                    lambda x: (
                        jnp.pad(
                            x[:max_grasps_per_part],
                            ((0, max(0, max_grasps_per_part - x.shape[0])), (0, 0)),
                            mode="edge",
                        )
                    ),
                    grasps,
                )
                grasp_list.append(grasps)
            grasps = jax.tree.map(lambda *x: jnp.stack(x), *grasp_list)
            logger.info(f"Grasp generation took {time.time() - start:.2f} seconds.")
        else:
            grasps = None

        with jdc.copy_and_mutate(_self, validate=False) as _self:
            _self.single_hand_assignments = single_hand_assignments
            _self.bimanual_assignments = bimanual_assignments
            _self.grasps = grasps

        return _self

    def get_part(self, idx: int) -> RSRDObject:
        group_mask = self.group_labels == idx
        return RSRDObject(
            self.means[group_mask] - self.means[group_mask].mean(axis=0),
            self.quats[group_mask],
            self._scales[group_mask],
            self._colors[group_mask],
            self._opacities[group_mask],
            num_groups=1,
            timesteps=self.timesteps,
            group_labels=jnp.zeros((group_mask.sum(),)),
            init_p2o=jaxlie.SE3.identity().wxyz_xyz,
            part_deltas=self.part_deltas[:, idx, :][:, None, :],
            _ns_to_world_scale=self._ns_to_world_scale,
            grasps=(
                None if self.grasps is None else
                jax.tree.map(lambda x: x[idx], self.grasps)
            ),
        )

    @property
    def covariances(self) -> Float[Array, "N 3 3"]:
        Rs = jaxlie.SO3(self.quats).as_matrix()
        cov = (
            jnp.einsum(
                "nij,njk,nlk->nil",
                Rs,
                jnp.eye(3)[None, :, :] * jnp.exp(self._scales[:, None, :]) ** 2,
                Rs,
            )
            * self._ns_to_world_scale**2
        )

        # ensure that cov is positive definite
        cov = jnp.where(
            (jnp.linalg.eigvalsh(cov) > 0).all(axis=-1)[..., None, None],
            cov,
            jnp.eye(3)[None, :, :] * 1e-6,
        )

        assert jnp.all(jnp.linalg.eigvalsh(cov) > 0)
        return cov

    @property
    def colors(self) -> Float[Array, "N 3"]:
        return jnp.clip(self._colors, 0.0, 1.0)

    @property
    def opacities(self) -> Float[Array, "N 1"]:
        return jax.nn.sigmoid(self._opacities)

    def rank_parts_to_move_single(self, data: dict) -> tuple[int]:
        mat_left, mat_right = self._get_matrix_for_hand_assignments(data)
        sum_part_dist = (mat_left + mat_right).sum(axis=0)
        part_indices = sum_part_dist.argsort()
        return tuple(part_indices.tolist())

    def rank_parts_to_move_bimanual(self, data: dict) -> tuple[tuple[int, int]]:
        mat_left, mat_right = self._get_matrix_for_hand_assignments(data)

        num_groups = mat_left.shape[1]
        assignments = []
        for li in range(num_groups):
            for ri in range(num_groups):
                if li == ri:
                    continue
                dist = mat_left[:, li].sum() + mat_right[:, ri].sum()
                assignments.append((li, ri, dist))
        assignments.sort(key=lambda x: x[2])
        return tuple[tuple[int, int]]([(a[0], a[1]) for a in assignments])

    def _get_matrix_for_hand_assignments(self, data: dict) -> list[onp.ndarray]:
        hands_info = data["hands"]

        num_timesteps = len(hands_info)
        num_parts = max(self.group_labels) + 1
        sum_dist_parts = [
            onp.zeros((num_timesteps, num_parts)),
            onp.zeros((num_timesteps, num_parts)),
        ]

        for timestep, (l_hand, r_hand) in hands_info.items():
            tstep = int(timestep)

            for part_idx in range(num_parts):
                delta = self.part_deltas[tstep, part_idx]
                part = self.get_part(part_idx)
                part_means = part.means
                part_means -= part_means.mean(axis=0)
                part_means = (
                    jaxlie.SE3(jnp.array(self.init_p2o[part_idx]))
                    @ jaxlie.SE3(jnp.array(delta))
                    @ jnp.array(part_means)
                )

                for hand_idx, hand in enumerate([l_hand, r_hand]):
                    if hand is None:
                        continue

                    keypoints = jnp.array(hand["keypoints_3d"])
                    if keypoints.shape[0] == 0:
                        continue

                    pointer = keypoints[:, 8] # [n_hands, 3]
                    thumb = keypoints[:, 4] # [n_hands, 3]
                    
                    pointer = jnp.expand_dims(pointer, axis=1) # [n_hands, 1, 3]
                    thumb = jnp.expand_dims(thumb, axis=1) # [n_hands, 1, 3]

                    # Scale by nerfstudio scale.
                    pointer *= self._ns_to_world_scale
                    thumb *= self._ns_to_world_scale

                    d_pointer = (
                        jnp.linalg.norm(pointer - part_means, axis=-1).min().item()
                    )
                    d_thumb = jnp.linalg.norm(thumb - part_means, axis=-1).min().item()
                    sum_dist_parts[hand_idx][tstep, part_idx] = (
                        d_pointer + d_thumb
                    ) / 2

        return sum_dist_parts


class RSRDVisualizer:
    """
    Visualization for RSRD. The frames are nested as: `/object/group_0/delta/gaussians`
    """

    _server: viser.ViserServer
    _scale: float
    rsrd_obj: RSRDObject
    part_delta_frames: list[viser.FrameHandle]
    num_groups: int

    def __init__(
        self,
        server: viser.ViserServer,
        rsrd_obj: RSRDObject,
        base_frame_name: str = "/object",
    ):
        self._server = server
        self.rsrd_obj = rsrd_obj

        self.part_delta_frames = []
        self.num_groups = int(rsrd_obj.group_labels.max() + 1)
        for group_idx in range(self.num_groups):
            self._create_frames_and_gaussians(
                group_idx,
                base_frame_name + "/group_" + str(group_idx),
            )

    def _create_frames_and_gaussians(self, group_idx: int, frame_name: str):
        group_rsrd = self.rsrd_obj.get_part(group_idx)
        init_p2o = self.rsrd_obj.init_p2o[group_idx]

        self._server.scene.add_frame(
            frame_name,
            position=onp.array(init_p2o[4:]),
            wxyz=onp.array(init_p2o[:4]),
            show_axes=False,
        )
        self.part_delta_frames.append(
            self._server.scene.add_frame(frame_name + "/delta", show_axes=False)
        )
        self._server.scene.add_gaussian_splats(
            frame_name + "/delta/gaussians",
            centers=onp.array(group_rsrd.means),
            rgbs=onp.array(group_rsrd.colors),
            opacities=onp.array(group_rsrd.opacities),
            covariances=onp.array(group_rsrd.covariances),
        )

    def update_cfg(self, timestep: int):
        """Update the configuration of the objects in the scene."""
        for group_idx in range(self.num_groups):
            group_delta = self.rsrd_obj.part_deltas[timestep, group_idx]
            self.part_delta_frames[group_idx].position = onp.array(
                jaxlie.SE3(group_delta).translation()
            )
            self.part_delta_frames[group_idx].wxyz = onp.array(
                jaxlie.SE3(group_delta).rotation().wxyz
            )

    def get_part_frame_name(self, group_idx: int) -> str:
        return f"/object/group_{group_idx}/delta/"