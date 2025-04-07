"""
Loads a robot from a URDF file or a robot description, using `yourdfpy`.
"""

from pathlib import Path
from typing import Optional

import yourdfpy

from ._robot import Robot


def load_robot(
    robot_urdf_path: Optional[Path] = None, robot_description: Optional[str] = None
) -> tuple[yourdfpy.URDF, Robot]:
    """
    Loads a robot from a URDF file or a robot description, using `yourdfpy`.

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
