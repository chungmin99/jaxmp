"""05_path_planning.py
Uses geometric path planning to find a path from start to goal.
"""

from __future__ import annotations

from typing import Optional
from jaxmp.coll._collide_types import Capsule
import viser
from viser.extras import ViserUrdf

import jax
import jax.numpy as jnp
import numpy as onp
import jaxlie
import jax_dataclasses as jdc

from jaxmp import JaxKinTree, RobotFactors
from jaxmp.coll import RobotColl
from jaxmp.extras import load_urdf, solve_ik

from jaxmp.path._mcts_types import NodeState, NodeTransition, Search


@jdc.pytree_dataclass
class JointState(NodeState[jnp.ndarray]):
    def get_value(self, target: NodeState[jax.Array]) -> jax.Array:
        return -jnp.linalg.norm(self.value - target.value)[None]

    def apply_action(
        self, action: jnp.ndarray, rng_key: Optional[jax.Array] = None
    ) -> NodeState[jnp.ndarray]:
        return JointState(self.value + action)


@jdc.pytree_dataclass
class JointTransition(NodeTransition[jnp.ndarray]):
    target_state: NodeState[jnp.ndarray]

    @staticmethod
    def from_actions_and_target(
        actions: jnp.ndarray, target_state: NodeState[jnp.ndarray]
    ) -> JointTransition:
        return JointTransition(actions.shape[0], actions, target_state)

    def get_transition(
        self, state: NodeState[jnp.ndarray], action_idx: jax.Array
    ) -> jnp.ndarray:
        dist_from_target = jnp.abs(self.target_state.value - state.value)
        return self.actions[action_idx] * dist_from_target
    
    def get_reward(
        self, state: NodeState[jnp.ndarray], state_next: NodeState[jnp.ndarray], target: NodeState[jnp.ndarray]
    ) -> jnp.ndarray:
        return state_next.get_value(target) - state.get_value(target)


def main():
    urdf = load_urdf("panda")
    kin = JaxKinTree.from_urdf(urdf)
    coll = RobotColl.from_urdf(urdf)
    rest_joints = (kin.limits_upper + kin.limits_lower) / 2
    JointVar = RobotFactors.get_var_class(kin)
    ik_weight = jnp.array([5.0] * 3 + [1.0] * 3)

    # Initialize robots from the rest joints.
    curr_joints = rest_joints

    server = viser.ViserServer()

    # Robot being moved with IK.
    urdf_vis = ViserUrdf(server, urdf)
    urdf_vis.update_cfg(onp.array(curr_joints))

    # Robot following with path planning.
    urdf_vis_path = ViserUrdf(server, urdf, root_node_name="/path")
    urdf_vis_path.update_cfg(onp.array(curr_joints))

    # GUI elements to control IK elements.
    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0],
    )
    target_tf_handle = server.scene.add_transform_controls(
        "target_transform", scale=0.2
    )

    sphere_obs = Capsule.from_radius_and_height(
        jnp.array([0.05]), jnp.array([2.0]), jaxlie.SE3.identity()
    )
    # sphere_obs = Sphere.from_center_and_radius(jnp.zeros(3), jnp.array([0.05]))
    sphere_obs_handle = server.scene.add_transform_controls(
        "sphere_obs", scale=0.2, position=(0.2, 0.0, 0.2)
    )
    server.scene.add_mesh_trimesh("sphere_obs/mesh", sphere_obs.to_trimesh())

    target_joints = rest_joints
    while True:
        n_actions = 30

        # Get the target joints.
        target_joint_indices = jnp.array(
            [kin.joint_names.index(target_name_handle.value)]
        )
        target_poses = jaxlie.SE3(
            jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position])
        )
        _, target_joints = solve_ik(
            kin, target_poses, target_joint_indices, target_joints, JointVar, ik_weight
        )
        urdf_vis.update_cfg(onp.array(target_joints))

        curr_sphere_obs = sphere_obs.transform(
            jaxlie.SE3(
                jnp.array([*sphere_obs_handle.wxyz, *sphere_obs_handle.position])
            )
        )

        start_state = JointState(curr_joints[None, ...])
        target_state = JointState(target_joints[None, ...])

        prng_key = jax.random.PRNGKey(0)
        actions = jax.random.uniform(
            prng_key, (n_actions, *curr_joints.shape), minval=-1, maxval=1
        )
        transitions = JointTransition.from_actions_and_target(actions, target_state)

        action = Search.solve(
            start_state,
            target_state,
            transitions,
            prng_key,
            max_depth=10,
            num_simulations=100,
        )
        print(action)

        curr_joints = start_state.apply_action(action).value[0]
        print(curr_joints)
        urdf_vis_path.update_cfg(onp.array(curr_joints))


if __name__ == "__main__":
    main()
