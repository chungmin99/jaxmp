import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pytest

import pyroki as pk
from pyroki._costs import (
    PoseCost,
)

# Ideally we can use this to test mimic joints, and that we're correctly tracking the link poses, etc.

@pytest.fixture
def simple_robot() -> pk.Robot:
    """Create a simple 2-DOF robot for testing."""
    return pk.load_robot(robot_description="double_pendulum")[1]


def test_pose_cost(simple_robot):
    """Test pose cost with simple cases."""
    joint_var = simple_robot.JointVar(0)
    cost = PoseCost.make(simple_robot, joint_var, jaxlie.SE3.identity(), jnp.array([1]))

    # Test identity case -- pose cost should be zero.
    vals = jaxls.VarValues.make([joint_var.with_value(jnp.zeros(2))])
    target = jaxlie.SE3.from_translation(jnp.array([0.0290872, 0.0, 0.135]))
    residual = cost.cost_fn(vals, simple_robot, joint_var, target, jnp.array([1]))
    assert jnp.allclose(residual, jnp.zeros(6))

    # Test with offset -- pose cost should be non-zero.
    vals = jaxls.VarValues.make([joint_var.with_value(jnp.array([0.5, 0.0]))])
    residual = cost.cost_fn(vals, simple_robot, joint_var, target, jnp.array([1]))
    assert not jnp.allclose(residual, jnp.zeros(6))
