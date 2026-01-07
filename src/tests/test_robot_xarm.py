# tests/test_robot_xarm.py
import types
from unittest.mock import MagicMock
import pytest

def test_move_gripper_success(fake_rospy, monkeypatch):
    # Import after rospy is stubbed
    from vr_pipeline.robot.xarm import RobotXArm, XArmServiceNames

    srv = XArmServiceNames(
        gripper_move="/xarm/gripper_move",
        move_joint="/xarm/move_joint",
        move_pose="/xarm/move_line",
        set_ee_twist="/xarm/vc_set_cartesian_velocity",
        stop="/xarm/motion_stop",
        home="/xarm/go_home",
    )
    robot = RobotXArm(srv=srv, gripper_min=-100, gripper_max=850)

    # Fake service response
    fake_res = types.SimpleNamespace(ret=0, message="ok")
    fake_proxy = MagicMock(return_value=fake_res)
    fake_rospy.ServiceProxy.return_value = fake_proxy

    rc = robot.move_gripper(100)
    assert rc == 0
    fake_rospy.wait_for_service.assert_called_with("/xarm/gripper_move")
    assert fake_proxy.called


def test_move_gripper_failure_ret(fake_rospy):
    from vr_pipeline.robot.xarm import RobotXArm, XArmServiceNames

    srv = XArmServiceNames(
        gripper_move="/xarm/gripper_move",
        move_joint="",
        move_pose="",
        set_ee_twist="",
        stop="",
        home="",
    )
    robot = RobotXArm(srv=srv, gripper_min=-100, gripper_max=850)

    fake_res = types.SimpleNamespace(ret=12, message="bad")
    fake_proxy = MagicMock(return_value=fake_res)
    fake_rospy.ServiceProxy.return_value = fake_proxy

    rc = robot.move_gripper(200)
    assert rc == -1


def test_move_gripper_out_of_range_raises(fake_rospy):
    from vr_pipeline.robot.xarm import RobotXArm, XArmServiceNames

    srv = XArmServiceNames(
        gripper_move="/xarm/gripper_move",
        move_joint="",
        move_pose="",
        set_ee_twist="",
        stop="",
        home="",
    )
    robot = RobotXArm(srv=srv, gripper_min=-100, gripper_max=850)

    with pytest.raises(AssertionError):
        robot.move_gripper(9999)
