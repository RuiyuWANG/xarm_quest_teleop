from __future__ import annotations

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _share_path(*parts: str) -> str:
    return str(Path(get_package_share_directory("xarm_quest_teleop")).joinpath(*parts))


def _resolve_config_path(path_value: str) -> str:
    raw = Path(os.path.expanduser(str(path_value)))
    if raw.is_absolute():
        return str(raw)
    return _share_path(str(raw))


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return data


def _camera_launch(camera_cfg: dict) -> IncludeLaunchDescription:
    launch_args = {}
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
        if key not in camera_cfg:
            continue
        value = camera_cfg[key]
        if isinstance(value, bool):
            value = str(value).lower()
        launch_args[key] = str(value)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])
        ),
        launch_arguments=launch_args.items(),
    )


def _launch_setup(context, *_args, **_kwargs):
    hardware_path = LaunchConfiguration("hardware_config").perform(context)
    camera_profile = LaunchConfiguration("camera_profile").perform(context)
    launch_robot = _truthy(LaunchConfiguration("launch_robot").perform(context))
    launch_quest = _truthy(LaunchConfiguration("launch_quest").perform(context))
    launch_cameras = _truthy(LaunchConfiguration("launch_cameras").perform(context))
    launch_stamp = _truthy(LaunchConfiguration("launch_stamp").perform(context))

    hw = _load_yaml(hardware_path)
    actions = []

    if launch_robot:
        robot = hw.get("robot", {})
        extra_path = _resolve_config_path(robot.get("extra_robot_api_params_path", "config/ros2/xarm_api_services.yaml"))
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([FindPackageShare("xarm_api"), "launch", "xarm7_driver.launch.py"])
                ),
                launch_arguments={
                    "robot_ip": str(robot.get("robot_ip", "192.168.1.243")),
                    "report_type": str(robot.get("report_type", "dev")),
                    "hw_ns": str(robot.get("hw_ns", "xarm")),
                    "add_gripper": str(bool(robot.get("add_gripper", True))).lower(),
                    "extra_robot_api_params_path": extra_path,
                }.items(),
            )
        )

    if launch_quest:
        quest = hw.get("quest", {})
        actions.append(
            Node(
                package="ros_tcp_endpoint",
                executable="default_server_endpoint",
                name="xarm_quest_ros_tcp_endpoint",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {"ROS_IP": str(quest.get("tcp_ip", "192.168.0.181"))},
                    {"ROS_TCP_PORT": int(quest.get("tcp_port", 10000))},
                ],
            )
        )

    if launch_stamp:
        actions.append(
            Node(
                package="xarm_quest_teleop",
                executable="quest_stamped_node",
                name="quest_stamp_node",
                output="screen",
                emulate_tty=True,
            )
        )

    if launch_cameras:
        cameras = hw.get("cameras", {})
        profile = hw.get("profiles", {}).get(camera_profile, {})
        for name in profile.get("cameras", []):
            cfg = cameras.get(name)
            if cfg is not None:
                actions.append(_camera_launch(cfg))

    return actions


def generate_launch_description():
    default_hardware = _share_path("config", "ros2", "hardware.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_config", default_value=default_hardware),
            DeclareLaunchArgument("camera_profile", default_value="rgb"),
            DeclareLaunchArgument("launch_robot", default_value="true"),
            DeclareLaunchArgument("launch_quest", default_value="true"),
            DeclareLaunchArgument("launch_cameras", default_value="true"),
            DeclareLaunchArgument("launch_stamp", default_value="true"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
