"""
Loads a robot from a URDF file or a robot description, using `yourdfpy`.
"""

from copy import deepcopy
from pathlib import Path
from typing import Optional

import yourdfpy

from loguru import logger

from ._robot import Robot


def load_robot(
    robot_urdf_path: Optional[Path] = None, robot_description: Optional[str] = None
) -> tuple[yourdfpy.URDF, Robot]:
    """
    Loads a robot from a URDF file or a robot description, using `yourdfpy`.
    Prior to loading, the URDF joints are sorted in topological order.

    Returns:
        - `urdf`: The loaded URDF.
        - `robot`: The robot object.
    """
    if robot_urdf_path is not None:
        urdf = _load_urdf(robot_urdf_path)
    elif robot_description is not None:
        urdf = _load_robot_description(robot_description)
    else:
        raise ValueError(
            "Either robot_urdf_path or robot_description must be provided."
        )

    urdf = _sort_joint_map(urdf)
    logger.info("TODO make sure robot takes in this sorted order!")
    return urdf, Robot.from_urdf(urdf)


def _load_urdf(robot_urdf_path: Path) -> yourdfpy.URDF:
    """
    Loads a robot from a URDF file, using yourdfpy.

    Applies two small changes:
    - Modifies yourdfpy filehandler to load files relative to the URDF file, and
    - Sorts the joints in topological order.
    """

    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)

    return yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)


def _load_robot_description(robot_description: str) -> yourdfpy.URDF:
    """
    Loads a robot from `robot_description`, using yourdfpy.
    """
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    if "description" not in robot_description:
        robot_description += "_description"
    return load_robot_description(robot_description)


def _sort_joint_map(urdf: yourdfpy.URDF) -> yourdfpy.URDF:
    """Return a sorted robot, with the joint map in topological order."""
    joints = deepcopy(urdf.robot.joints)

    # Sort the joints in topological order.
    sorted_joints = list[yourdfpy.Joint]()
    joint_from_child = {j.child: j for j in joints}
    while joints:
        for j in joints:
            if j.parent not in joint_from_child:
                sorted_joints.append(j)
                joints.remove(j)
                joint_from_child.pop(j.child)
                break
        else:
            raise ValueError("Cycle detected in URDF!")

    # Update the joints.
    robot = deepcopy(urdf.robot)
    robot.joints = sorted_joints

    # Re-load urdf, with the updated robot.
    filename_handler = urdf._filename_handler  # pylint: disable=protected-access
    updated_urdf = yourdfpy.URDF(
        robot=robot,
        filename_handler=filename_handler,
    )
    return updated_urdf
