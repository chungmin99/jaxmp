"""05_path_planning.py
Uses geometric path planning to find a path from start to goal.
"""

from __future__ import annotations

from typing import Optional
from jaxmp.coll._collide import collide
from jaxmp.coll._collide_types import Capsule, CollGeom, Sphere
import viser
from viser.extras import ViserUrdf

import jax
import jax.numpy as jnp
import numpy as onp
import jaxlie
import jax_dataclasses as jdc

from jaxmp import JaxKinTree, RobotFactors
from jaxmp.coll import RobotColl, link_to_spheres
from jaxmp.extras import load_urdf, solve_ik

from jaxmp.path._mcts_types import NodeState, NodeTransition, Search


@jdc.pytree_dataclass
class JointState(NodeState[jnp.ndarray]):
    def get_value(self, target: NodeState[jax.Array]) -> jax.Array:
        # return -jnp.linalg.norm(self.value - target.value)[None]
        return -jnp.abs(self.value - target.value).sum()[None]

    def apply_action(
        self, action: jnp.ndarray, rng_key: Optional[jax.Array] = None
    ) -> NodeState[jnp.ndarray]:
        return JointState(self.value + action)


@jdc.pytree_dataclass
class JointTransition(NodeTransition[jnp.ndarray]):
    target_state: NodeState[jnp.ndarray]
    kin: JaxKinTree
    coll: RobotColl
    world_coll: CollGeom
    target_joint_idx: int

    @staticmethod
    def from_actions_and_target(
        actions: jnp.ndarray, target_state: NodeState[jnp.ndarray], kin, coll, world_coll, target_joint_idx
    ) -> JointTransition:
        return JointTransition(actions.shape[0], actions, target_state, kin, coll, world_coll, target_joint_idx)

    def get_transition(
        self, state: NodeState[jnp.ndarray], action_idx: jax.Array
    ) -> jnp.ndarray:
        action = self.actions[action_idx]
        # t = 0.1
        # return action * self.kin.joint_vel_limit * 0.1
        # action = jnp.clip(action, -self.kin.joint_vel_limit * t, self.kin.joint_vel_limit * t)
        dist_from_target = jnp.abs(self.target_state.value - state.value)
        return action * dist_from_target
    
    def get_transition_outputs(
        self, state: NodeState[jnp.ndarray], action: jnp.ndarray, target: NodeState[jnp.ndarray]
    ) -> tuple[NodeState[jnp.ndarray], jax.Array, jax.Array]:
        # Return next state, value, and reward.
        # Check collision _at_position
        state_next = state.apply_action(action)
        _curr_coll = self.coll.at_joints(self.kin, state.value)
        _next_coll = self.coll.at_joints(self.kin, state_next.value)
        assert isinstance(_curr_coll, Sphere)
        assert isinstance(_next_coll, Sphere)

        value = (-jnp.linalg.norm(state_next.value - target.value))[None]
        value_prev = (-jnp.linalg.norm(state.value - target.value))[None]
        # curr_pose = self.kin.forward_kinematics(state.value)[self.target_joint_idx]
        # next_pose = self.kin.forward_kinematics(state_next.value)[self.target_joint_idx]
        # target_pose = self.kin.forward_kinematics(target.value)[self.target_joint_idx]
        # value = -jnp.linalg.norm(jaxlie.SE3.log(jaxlie.SE3(next_pose) @ jaxlie.SE3(target_pose).inverse())).sum()[None]
        # value_prev = -jnp.linalg.norm(jaxlie.SE3.log(jaxlie.SE3(curr_pose) @ jaxlie.SE3(target_pose).inverse())).sum()[None]

        reward = (value - value_prev)

        coll_sweep = Capsule.from_sphere_pairs(_curr_coll, _next_coll)
        dist = collide(coll_sweep, self.world_coll).dist.min()
        coll_score = jnp.clip(100*(dist), min=-10.0, max=0.0)
        reward = reward + coll_score
        value = value + coll_score

        return JointState(state_next.value), value, reward
    
    def get_empty_transition(self) -> jax.Array:
        return jnp.zeros_like(self.actions[0])


def main():
    urdf = load_urdf("panda")
    kin = JaxKinTree.from_urdf(urdf)
    coll = RobotColl.from_urdf(urdf, create_coll_bodies=link_to_spheres)
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
        n_actions = 100

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
        transitions = JointTransition.from_actions_and_target(actions, target_state, kin, coll, curr_sphere_obs, target_joint_indices[0])

        action = Search.solve(
            start_state,
            target_state,
            transitions,
            prng_key,
            max_depth=10,
            num_simulations=100,
        )

        curr_joints = start_state.apply_action(action).value[0]
        urdf_vis_path.update_cfg(onp.array(curr_joints))


if __name__ == "__main__":
    main()
