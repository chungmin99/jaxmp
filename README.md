# `PyRoki`: Python Robot Kinematics Library

...

The only real dependency should be JAX

Modularity, extensibility, cross-platform

Core features include:
- Differentiable forward robot kinematics model, given a URDF from [`yourdfpy`](https://github.com/clemense/yourdfpy/tree/main) as input.
  - Supports a wide range of robots, through [robot-descriptions](https://github.com/robot-descriptions/robot_descriptions.py).
  - Automatic robot collision geometry generation (e.g., with capsules).
- Differentiable collision bodies with numpy broadcasting logic. 
- Common cost factors (e.g., EE pose, self/world-collision, manipulability).
- Arbitrary costs, as long as autodiff Jacobians are feasible.

Please refer to the [docs]() for more features and usage examples.


## Installation

Install with:

```
pip install pyroki
```

To run examples, install with `pip install -e .[examples]`.

## Quick start
### Defining objectives
Every `pyroki` code starts with loading a robot:

```python
import pyroki as pk
robot = pk.load_robot(robot_description="panda")
```

, and composing cost functions for some objective -- for instance, for global inverse kinematics.

```python
joints = robot.JointVar(0)

vars = [joints]
factors = [
  pk.PoseCost(robot, joints, target_pose),
  pk.LimitCost(robot, joints),
]
sol = pk.solve(vars, factors, *args)
```

### Applying modularity

One of the core design decisions behind `pyroki` is _modularity_: users should be able to
- re-use code across different tasks (global / differential IK, trajopt), and
- apply the same objectives (e.g., collision avoidance, pose matching) across tasks.

The previous global IK snippit can be easily modified for differential IK:

```
...
```

, or for trajectory optimization, to smoothly follow target poses from the end-effector frame.

```
...
```

We can easily modify this IK code for a mobile base, simultaneously solving for the base pose:

```
...
```

### Extending costs
Another core design choice is _extensibility_: it should be easy to add custom costs.

Let's add (cost).


## Acknowledgements
...
