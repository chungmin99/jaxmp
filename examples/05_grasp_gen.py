"""
05_grasp_gen.py
Antipodal grasp sampling demo.
"""
import time
from pathlib import Path

import tyro
import trimesh
import viser

import numpy as onp
import jax

from jaxmp.extras import AntipodalGrasps

def main(
    obj_path: Path = Path(__file__).parent / "assets/ycb_cracker_box.obj",
    num_samples: int = 1000,
    max_width: float = 0.04,
    max_angle_deviation: float = onp.pi / 8,
):
    obj_mesh = trimesh.load(obj_path)
    assert isinstance(obj_mesh, trimesh.Trimesh)
    
    # create jax prng key
    prng_key = jax.random.PRNGKey(0)

    grasps = AntipodalGrasps.from_sample_mesh(
        obj_mesh,
        prng_key=prng_key,
        max_samples=num_samples,
        max_width=max_width,
        max_angle_deviation=max_angle_deviation,
    )

    server = viser.ViserServer()
    server.scene.add_mesh_trimesh("obj/mesh", obj_mesh)
    server.scene.add_mesh_trimesh("obj/grasps", grasps.to_trimesh())

    # Maybe an area metric?

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)