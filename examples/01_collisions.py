"""01_collisions.py
Test collision detection between various pyroki geometry types using Viser.
Includes testing for batched geometries with unified batch control.
"""

from __future__ import annotations

import time
import trimesh
import trimesh.visual
from typing import cast, Dict

import jax.numpy as jnp
import jaxlie
import jax
import numpy as onp

import viser

from pyroki.coll import (
    collide,
    HalfSpace,
    Sphere,
    Capsule,
    Box,
    CollGeom,
)


def main():
    server = viser.ViserServer()
    timing_handle = server.gui.add_number("Timing (ms)", 0.001, disabled=True)

    # Helper to create identity transform as jaxlie.SE3
    def identity_pose(batch_axes=()) -> jaxlie.SE3:
        pos = jnp.zeros(batch_axes + (3,), dtype=jnp.float32)
        num_batch_elements = onp.prod(batch_axes).item() if batch_axes else 1
        quat_wxyz = jnp.stack(
            [jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)] * num_batch_elements,
            axis=0,
        )
        quat_wxyz = quat_wxyz.reshape(batch_axes + (4,))
        wxyz_xyz = jnp.concatenate([quat_wxyz, pos], axis=-1)
        return jaxlie.SE3(wxyz_xyz)

    # Helper to get pose from viser handle
    def pose_from_handle(handle: viser.TransformControlsHandle) -> jaxlie.SE3:
        return jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.array(handle.wxyz)), jnp.array(handle.position)
        )

    # --- Create Collision Geometries ---
    # HalfSpace (single)
    hs_pose = identity_pose()
    halfspace = HalfSpace(pose=hs_pose, size=jnp.zeros((1,)))  # Dummy size
    halfspace = halfspace.reshape(1)

    # Sphere (batched, size 2)
    sphere_batch_shape = (2,)
    sphere_centers = jnp.array([[0.0, 0.0, 0.5], [0.5, 0.0, 0.5]])  # Initial positions
    sphere_radius = jnp.array([0.1, 0.15])  # Different radii for batch elements
    sphere_pose_0 = jaxlie.SE3.from_translation(sphere_centers[0])
    sphere_pose_1 = jaxlie.SE3.from_translation(sphere_centers[1])
    stacked_sphere_params = jnp.stack(
        [sphere_pose_0.wxyz_xyz, sphere_pose_1.wxyz_xyz], axis=0
    )
    spheres_pose = jaxlie.SE3(stacked_sphere_params)
    spheres_size = sphere_radius[:, None]  # Shape (2, 1)
    spheres = Sphere(pose=spheres_pose, size=spheres_size)  # Already has batch dim 2

    # Capsule (single)
    cap_identity_pose = identity_pose()
    cap_pose = jaxlie.SE3.from_rotation_and_translation(
        cap_identity_pose.rotation(), jnp.array([-0.5, 0.0, 0.5])
    )
    cap_size = jnp.array([0.1, 0.2])  # Shape (2,) - Fixed to match Capsule definition
    capsule = Capsule(pose=cap_pose, size=cap_size)
    capsule = capsule.reshape(1)  # Add batch dim 1

    # Box (single)
    box_pose = identity_pose()
    box_pose = jaxlie.SE3.from_rotation_and_translation(
        box_pose.rotation(), jnp.array([0.5, 0.5, 0.25])  # Initial position
    )
    box_extents = jnp.array([0.5, 0.6, 0.7])  # lx, ly, lz
    box = Box(pose=box_pose, size=box_extents)
    box = box.reshape(1)  # Add batch dim

    # Store geometries and their names/prefixes
    geom_dict: Dict[str, CollGeom] = {
        "halfspace": halfspace,
        "spheres": spheres,
        "capsule": capsule,
        "box": box,
    }

    # --- Viser Setup ---
    # Create ONE handle per geometry object (even if batched)
    handles_dict: Dict[str, viser.TransformControlsHandle] = {}
    mesh_paths_dict: Dict[str, str] = {}
    # Store initial poses from handles to calculate deltas later
    previous_handle_poses: Dict[str, jaxlie.SE3] = {}

    for name, geom in geom_dict.items():
        # Use first element's pose for handle initialization
        batch_axes = geom.get_batch_axes()
        idx_tuple = tuple(
            [0] * len(batch_axes)
        )  # Index for first element (e.g., (0,) or (0,0))
        handle_initial_pose: jaxlie.SE3 = jax.tree.map(
            lambda x: x[idx_tuple], geom.pose
        )

        pos_init = tuple(onp.array(handle_initial_pose.translation()))
        quat_init = tuple(onp.array(handle_initial_pose.rotation().wxyz))

        # Create a single handle for the geometry object
        handle_path = f"/coll/{name}/control"  # Single control handle
        mesh_paths_dict[name] = f"/coll/{name}/mesh"  # Mesh path

        handle = server.scene.add_transform_controls(
            handle_path, wxyz=quat_init, position=pos_init
        )
        handles_dict[name] = handle
        # Store the initial pose read *from the handle*
        previous_handle_poses[name] = pose_from_handle(handle)

    # --- Main Loop ---
    # Store current state of geometries (updated each loop)
    current_geom_dict = geom_dict.copy()

    while True:
        start_time = time.time()
        next_geom_dict = {}  # Build the next state here

        # Update geometry poses based on handle deltas
        for name, geom in current_geom_dict.items():  # Use current state
            handle = handles_dict[name]
            prev_handle_pose = previous_handle_poses[name]
            current_handle_pose = pose_from_handle(handle)

            # Calculate delta transform: T_delta = T_current @ T_previous.inverse()
            delta_transform = current_handle_pose @ prev_handle_pose.inverse()

            # Apply delta transform to the entire geometry object (batched or not)
            next_geom_dict[name] = geom.transform(delta_transform)

            # Update previous handle pose for next iteration
            previous_handle_poses[name] = current_handle_pose

        # Update the current state for the next iteration
        current_geom_dict = next_geom_dict

        # Collision checking (using the updated geometries in next_geom_dict)
        geom_names = list(current_geom_dict.keys())
        in_collision_dict: Dict[str, bool] = {name: False for name in geom_names}

        for i in range(len(geom_names)):
            for j in range(i + 1, len(geom_names)):
                name1 = geom_names[i]
                name2 = geom_names[j]
                geom1 = current_geom_dict[name1]  # Use updated geoms
                geom2 = current_geom_dict[name2]  # Use updated geoms

                dist = collide(geom1, geom2)

                if dist is None or jnp.any(jnp.isnan(dist)):
                    continue

                try:
                    expected_shape = jnp.broadcast_shapes(
                        geom1.get_batch_axes(), geom2.get_batch_axes()
                    )
                    if dist.shape != expected_shape:
                        print(
                            f"Shape mismatch! {name1} ({geom1.get_batch_axes()}) <-> {name2} ({geom2.get_batch_axes()}). Expected {expected_shape}, got {dist.shape}"
                        )
                        continue
                except ValueError:
                    print(
                        f"Error broadcasting shapes for collision check: {name1} vs {name2}"
                    )
                    continue

                if jnp.any(dist < 0.0):
                    in_collision_dict[name1] = True
                    in_collision_dict[name2] = True

        timing_handle.value = (time.time() - start_time) * 1000

        # Visualize (using the updated geometries)
        for name, coll in current_geom_dict.items():
            mesh_path = mesh_paths_dict[name]
            in_collision = in_collision_dict[name]
            try:
                mesh = coll.to_trimesh()
                if mesh.is_empty:
                    continue

                if not hasattr(mesh, "visual") or not isinstance(
                    mesh.visual, trimesh.visual.ColorVisuals
                ):
                    mesh.visual = trimesh.visual.ColorVisuals()

                visuals: trimesh.visual.ColorVisuals = cast(
                    trimesh.visual.ColorVisuals, mesh.visual
                )

                color = (
                    onp.array([255, 100, 100, 200])
                    if in_collision
                    else onp.array([100, 255, 100, 200])
                )
                visuals.face_colors = color

                server.scene.add_mesh_trimesh(mesh_path, mesh)

            except Exception as e:
                print(
                    f"Error visualizing geometry {name} ({coll.__class__.__name__}): {e}"
                )

        time.sleep(0.01)


if __name__ == "__main__":
    main()
