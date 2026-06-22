from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - lets non-ROS unit tests run.
    get_package_share_directory = None


PACKAGE_NAME = "xarm_quest_teleop"


def package_share_dir() -> Path:
    if get_package_share_directory is not None:
        try:
            return Path(get_package_share_directory(PACKAGE_NAME))
        except Exception:
            pass
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return package_share_dir()


def resolve_repo_path(path: str | os.PathLike) -> str:
    p = Path(os.path.expanduser(str(path)))
    if p.is_absolute():
        return str(p)
    return str(repo_root() / p)


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    resolved = Path(resolve_repo_path(path))
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {resolved}")
    return data


def load_hardware_config(path: str = "config/ros2/hardware.yaml") -> Dict[str, Any]:
    return load_yaml(path)


def camera_topic(camera_name: str, stream: str) -> str:
    return f"/{camera_name}/{stream}/image_raw"


def camera_info_topic(camera_name: str, stream: str = "color") -> str:
    return f"/{camera_name}/{stream}/camera_info"


def realsense_launch_cmd(camera_cfg: Dict[str, Any]) -> List[str]:
    cmd = ["ros2", "launch", "realsense2_camera", "rs_launch.py"]
    for key in (
        "camera_name",
        "camera_namespace",
        "serial_no",
        "enable_color",
        "enable_depth",
        "rgb_camera.color_profile",
        "depth_module.depth_profile",
        "enable_sync",
        "align_depth.enable",
        "initial_reset",
    ):
        if key in camera_cfg:
            value = camera_cfg[key]
            if isinstance(value, bool):
                value = str(value).lower()
            cmd.append(f"{key}:={value}")
    return cmd


def camera_launch_cmds(profile: str = "rgb", hardware: Optional[Dict[str, Any]] = None) -> List[List[str]]:
    hw = load_hardware_config() if hardware is None else hardware
    names: Iterable[str] = hw.get("profiles", {}).get(profile, {}).get("cameras", [])
    cameras = hw.get("cameras", {})
    return [realsense_launch_cmd(cameras[name]) for name in names if name in cameras]


def xarm_launch_cmd(hardware: Optional[Dict[str, Any]] = None) -> List[str]:
    hw = load_hardware_config() if hardware is None else hardware
    robot = hw.get("robot", {})
    extra_path = resolve_repo_path(robot.get("extra_robot_api_params_path", "config/ros2/xarm_api_services.yaml"))
    return [
        "ros2",
        "launch",
        "xarm_api",
        "xarm7_driver.launch.py",
        f"robot_ip:={robot.get('robot_ip', '192.168.1.243')}",
        f"report_type:={robot.get('report_type', 'dev')}",
        f"hw_ns:={robot.get('hw_ns', 'xarm')}",
        f"add_gripper:={str(bool(robot.get('add_gripper', True))).lower()}",
        f"extra_robot_api_params_path:={extra_path}",
    ]


def quest_endpoint_cmd(hardware: Optional[Dict[str, Any]] = None) -> List[str]:
    hw = load_hardware_config() if hardware is None else hardware
    quest = hw.get("quest", {})
    return [
        "ros2",
        "run",
        "ros_tcp_endpoint",
        "default_server_endpoint",
        "--ros-args",
        "-p",
        f"ROS_IP:={quest.get('tcp_ip', '192.168.0.181')}",
        "-p",
        f"ROS_TCP_PORT:={int(quest.get('tcp_port', 10000))}",
    ]
