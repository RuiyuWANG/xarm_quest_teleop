#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
from typing import List, Optional

from xarm_quest_teleop.ros import compat as rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from xarm_quest_teleop.robots.xarm import XArmRobot
from xarm_quest_teleop.configs.robot_config import (
    HOME_JOINT,
    ROBOT_TOPIC,
    SRV_GO_HOME,
    SRV_GRIPPER_MOVE,
    SRV_GRIPPER_STATE,
    SRV_MOTION_ENABLE,
    SRV_MOVE_JOINT,
    SRV_MOVE_LINE,
    SRV_MOVE_SERVO_CART,
    SRV_SET_MODE,
    SRV_SET_STATE,
    SRV_VELO_MOVE_LINE_TIMED,
)
from xarm_quest_teleop.configs.ros2_config import xarm_launch_cmd


ROBOT_LAUNCH_CMD = xarm_launch_cmd()


REQUIRED_SERVICES = [
    SRV_SET_MODE,
    SRV_SET_STATE,
    SRV_MOTION_ENABLE,
    SRV_GO_HOME,
    SRV_MOVE_JOINT,
    SRV_MOVE_LINE,
    SRV_MOVE_SERVO_CART,
    SRV_VELO_MOVE_LINE_TIMED,
    SRV_GRIPPER_MOVE,
    SRV_GRIPPER_STATE,
]
REQUIRED_TOPICS = [
    ROBOT_TOPIC,
]


def wait_for_services(services: List[str], timeout_s: float = 20.0) -> None:
    start = time.time()
    remaining = set(services)
    while time.time() - start < timeout_s and remaining:
        ok_now = []
        for s in list(remaining):
            try:
                rospy.wait_for_service(s, timeout=0.2)
                ok_now.append(s)
            except Exception:
                pass
        for s in ok_now:
            remaining.discard(s)
        time.sleep(0.1)
    if remaining:
        raise RuntimeError(f"Timed out waiting for services: {sorted(remaining)}")


def wait_for_topics(topics: List[str], timeout_s: float = 20.0) -> None:
    start = time.time()
    remaining = set(topics)
    while time.time() - start < timeout_s and remaining:
        published = set([t for (t, _type) in rospy.get_published_topics()])
        for t in list(remaining):
            if t in published:
                remaining.discard(t)
        time.sleep(0.2)
    if remaining:
        raise RuntimeError(f"Timed out waiting for topics: {sorted(remaining)}")


def launch_robot_node() -> subprocess.Popen:
    """
    Start robot bringup using ROS2 launch.
    """
    env = os.environ.copy()
    p = subprocess.Popen(
        ROBOT_LAUNCH_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,  # so we can kill the whole process group
        text=True,
        bufsize=1,
    )
    return p


def shutdown_launch(p: subprocess.Popen) -> None:
    if p is None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except Exception:
        pass
    try:
        p.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass

def assert_ok(name: str, res) -> None:
    """
    res is expected to be CallResult-like with .ok/.ret/.message
    """
    if not getattr(res, "ok", False):
        raise RuntimeError(f"[{name}] failed: ret={getattr(res, 'ret', None)} msg={getattr(res, 'message', '')}")


def main():
    # IMPORTANT: This script assumes you run it in a terminal where:
    #   source /opt/ros/humble/setup.bash
    #   source ~/ros2_ws/install/setup.bash
    #
    # It can start the robot bringup itself, but the ROS2 env must be present.

    rospy.init_node("xarm_robot_node_test", anonymous=True)

    p = None
    try:
        print("[test] launching robot bringup:", " ".join(ROBOT_LAUNCH_CMD))
        p = launch_robot_node()

        # Wait for endpoints
        print("[test] waiting for topics/services ...")
        wait_for_topics(REQUIRED_TOPICS, timeout_s=30.0)
        wait_for_services(REQUIRED_SERVICES, timeout_s=30.0)

        # Create robot wrapper (blocking)
        robot = XArmRobot(auto_init=False)

        # Wait for state
        robot.wait_for_state(timeout_s=10.0)
        print("[test] robot state OK")

        # 1) Initialization
        print("[test] 1) initialization")
        assert_ok("set_mode(0)", robot.set_mode(0))
        assert_ok("set_state(0)", robot.set_state(0))
        
        # Open gripper as part of init (optional but matches your idea of init gripper)
        assert_ok("move_gripper(open)", robot.move_gripper(500))

        # 2) Homing
        print("[test] 2) home")
        assert_ok("home()", robot.home())

        # 3a) Joint control
        print("[test] 3a) move_to_joint then home")
        # a small offset from home
        joint_a = HOME_JOINT.copy()
        joint_a[0] += 0.10  # rad
        assert_ok("move_to_joint(joint_a)", robot.move_to_joint(joint_a, mvvelo=0, mvacc=0, mvtime=0))
        assert_ok("home()", robot.home())

        # 3b) Pose control (move_line)
        print("[test] 3b) move_to_pose then home")
        # Use current pose and do a tiny delta in z, if pose is available.
        # Your RobotMsg.pose is 6: [x_mm, y_mm, z_mm, r, p, y]
        pose6 = robot.get_state().ee_pose
        pose6 = list(pose6)
        pose6[2] = pose6[2] + 10.0  # +10mm
        assert_ok("move_to_pose(+z)", robot.move_to_pose(pose6, mvvelo=0, mvacc=0, mvtime=0))
        assert_ok("home()", robot.home())

        # 3c) TCP velocity control (velo_move_line_timed)
        print("[test] 3c) velo_move_line_timed then home")
        # This expects xarm_msgs/MoveVelocity (your earlier check).
        # Typical field is a 6D velocity + time. We'll command small +Z for 0.5s.
        # Units depend on driver; start very small.
        v = [0.2,0.2,0.2,0,0,0]  # (likely mm/s; adjust if needed)
        assert_ok("velo_move_line_timed", robot.velo_move_line_timed(v, duration=1/60))
        rospy.sleep(2.0) # WAIT for the velocity to actually happen
        assert_ok("home()", robot.home())

        print("\n[test] ALL PASSED ✅")

    finally:
        print("[test] shutting down bringup")
        shutdown_launch(p)


if __name__ == "__main__":
    main()
