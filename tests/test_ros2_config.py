from __future__ import annotations

from xarm_quest_teleop.configs.ros2_config import camera_launch_cmds, load_hardware_config, xarm_launch_cmd


def test_hardware_config_loads_ros2_defaults():
    hw = load_hardware_config()

    assert hw["robot"]["dof"] == 7
    assert "d405" in hw["cameras"]
    assert hw["profiles"]["rgb"]["cameras"] == ["d405", "d435i_front"]


def test_launch_commands_are_ros2_native():
    hw = load_hardware_config()

    xarm_cmd = xarm_launch_cmd(hw)
    camera_cmd = camera_launch_cmds("rgb", hw)[0]

    assert xarm_cmd[:3] == ["ros2", "launch", "xarm_api"]
    assert "xarm7_driver.launch.py" in xarm_cmd
    assert camera_cmd[:3] == ["ros2", "launch", "realsense2_camera"]
    assert "rs_launch.py" in camera_cmd
