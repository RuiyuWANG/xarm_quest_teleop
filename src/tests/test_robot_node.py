#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
from typing import List, Optional

import rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.robots.xarm import XArmRobot
from src.configs.robot_config import HOME_JOINT


# --------- USER: set these to your actual bringup launch ----------
# Example:
#   roslaunch xarm_bringup xarm7_bringup.launch robot_ip:=192.168.1.123
ROBOT_LAUNCH_CMD = [
    "roslaunch",
    "xarm_bringup",
    "xarm7_server.launch",
    "robot_ip:=192.168.1.241",
    "report_type:=dev",
    "add_gripper:=true",
]
# ---------------------------------------------------------------


REQUIRED_SERVICES = [
    "/xarm/set_mode",
    "/xarm/set_state",
    "/xarm/go_home",
    "/xarm/move_joint",
    "/xarm/move_line",
    "/xarm/velo_move_line_timed",
    "/xarm/gripper_move",
    "/xarm/gripper_state",
]
REQUIRED_TOPICS = [
    "/xarm/xarm_states",
]


def wait_for_ros_master(timeout_s: float = 10.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            import rosgraph
            if rosgraph.is_master_online():
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("ROS master is not online. Is roscore/roslaunch running?")


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
    Start robot bringup using roslaunch.
    """
    env = os.environ.copy()
    # Ensure python uses correct ROS env if needed
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
    #   source /opt/ros/noetic/setup.bash
    #   source ~/catkin_ws/devel/setup.bash
    #   export PYTHONPATH=... (repo)
    #
    # It can start the robot bringup itself, but ROS env must be present.

    rospy.init_node("xarm_robot_node_test", anonymous=True)

    p = None
    try:
        print("[test] launching robot bringup:", " ".join(ROBOT_LAUNCH_CMD))
        p = launch_robot_node()

        # Wait for ROS master to come online (roslaunch starts one if needed)
        wait_for_ros_master(timeout_s=10.0)

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
        assert_ok("move_gripper(open)", robot.move_gripper(800))

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
        v = [30,0,0,0,0,0]  # (likely mm/s; adjust if needed)
        assert_ok("velo_move_line_timed", robot.velo_move_line_timed(v, duration=3))
        rospy.sleep(2.0) # WAIT for the velocity to actually happen
        assert_ok("home()", robot.home())

        print("\n[test] ALL PASSED ✅")

    finally:
        print("[test] shutting down bringup")
        shutdown_launch(p)


if __name__ == "__main__":
    main()
