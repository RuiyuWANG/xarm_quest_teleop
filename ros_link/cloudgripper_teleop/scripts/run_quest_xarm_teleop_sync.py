#!/usr/bin/env python3
import os
import signal
import subprocess
import threading
import time
import sys
from typing import Optional, List

import rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.configs.teleop_config import TeleopConfig
from src.io.quest2 import Quest2Interface
from src.robots.xarm import XArmRobot
from src.teleop.quest_xarm_teleop_sync import QuestXArmTeleopSync
from src.configs.robot_config import ROBOT_TOPIC, XArmServices


class ManagedProcess:
    def __init__(self, name: str, cmd: List[str], workdir: Optional[str], pipe_output: bool):
        self.name = name
        self.cmd = cmd
        self.workdir = workdir
        self.pipe_output = pipe_output
        self.p: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        env = os.environ.copy()
        self.p = subprocess.Popen(
            self.cmd,
            cwd=self.workdir,
            stdout=subprocess.PIPE if self.pipe_output else None,
            stderr=subprocess.STDOUT if self.pipe_output else None,
            env=env,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1,
        )
        rospy.loginfo(f"[launch] started {self.name}: {' '.join(self.cmd)} (pid={self.p.pid})")
        if self.pipe_output and self.p.stdout is not None:
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()

    def _pump(self):
        assert self.p is not None and self.p.stdout is not None
        for line in self.p.stdout:
            line = line.rstrip("\n")
            if line:
                rospy.loginfo_throttle(0.2, f"[{self.name}] {line}")

    def alive(self) -> bool:
        return self.p is not None and self.p.poll() is None

    def stop(self):
        if self.p is None or self.p.poll() is not None:
            return
        try:
            pgid = os.getpgid(self.p.pid)
            rospy.logwarn(f"[launch] stopping {self.name} pgid={pgid}")
            os.killpg(pgid, signal.SIGINT)
        except Exception as e:
            rospy.logwarn(f"[launch] stop {self.name} failed: {e}")

        t0 = time.time()
        while time.time() - t0 < 5.0 and self.alive():
            time.sleep(0.05)
        if self.alive():
            try:
                pgid = os.getpgid(self.p.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass


def wait_for_topic(topic: str, timeout_s: float) -> bool:
    t0 = rospy.Time.now().to_sec()
    r = rospy.Rate(20)
    while not rospy.is_shutdown():
        pubs = rospy.get_published_topics()
        if any(t[0] == topic for t in pubs):
            return True
        if rospy.Time.now().to_sec() - t0 > timeout_s:
            return False
        r.sleep()
    return False


def wait_for_service(name: str, timeout_s: float) -> bool:
    try:
        rospy.wait_for_service(name, timeout=timeout_s)
        return True
    except Exception:
        return False


def main():
    rospy.init_node("quest_xarm_teleop_sync", anonymous=False)

    cfg = TeleopConfig()
    services = XArmServices()

    procs: List[ManagedProcess] = []

    def shutdown():
        rospy.logwarn("[main] shutting down, stopping launched processes")
        for p in reversed(procs):
            try:
                p.stop()
            except Exception:
                pass

    rospy.on_shutdown(shutdown)

    # autolaunch quest + robot
    if cfg.auto_launch_quest:
        p = ManagedProcess("quest_ros_tcp_endpoint", cfg.QUEST_LAUNCH_CMD, cfg.launch_workdir, cfg.pipe_launch_output)
        p.start()
        procs.append(p)

    if cfg.auto_launch_robot:
        p = ManagedProcess("xarm_bringup", cfg.ROBOT_LAUNCH_CMD, cfg.launch_workdir, cfg.pipe_launch_output)
        p.start()
        procs.append(p)

    # start wrapper nodes (as rosrun)
    # assumes scripts are executable and in your package
    stamp_quest_cmd = ["rosrun", cfg.scripts_package, "quest_stamp_node.py"]
    stamp_robot_cmd = ["rosrun", cfg.scripts_package, "robot_state_stamp_node.py"]

    p = ManagedProcess("quest_stamp_node", stamp_quest_cmd, cfg.launch_workdir, cfg.pipe_launch_output)
    p.start()
    procs.append(p)

    p = ManagedProcess("robot_state_stamp_node", stamp_robot_cmd, cfg.launch_workdir, cfg.pipe_launch_output)
    p.start()
    procs.append(p)

    # wait for stamped topics + xarm services
    rospy.loginfo("[startup] waiting for stamped topics/services...")

    if cfg.active_hand == "right":
        pose_t = cfg.right_pose_topic
        twist_t = cfg.right_twist_stamped_topic
        inputs_t = cfg.right_inputs_stamped_topic
    else:
        pose_t = cfg.left_pose_topic
        twist_t = cfg.left_twist_stamped_topic
        inputs_t = cfg.left_inputs_stamped_topic

    must_topics = [pose_t, twist_t, inputs_t, cfg.robot_state_stamped_topic]
    for t in must_topics:
        if not wait_for_topic(t, cfg.startup_timeout_s):
            rospy.logerr(f"[startup] missing topic: {t}")
            raise SystemExit(1)

    if not wait_for_topic(ROBOT_TOPIC, cfg.startup_timeout_s):
        rospy.logerr(f"[startup] missing robot topic: {ROBOT_TOPIC}")
        raise SystemExit(1)

    must_srvs = [services.set_mode, services.set_state, services.velo_move_line_timed,
                 services.gripper_move, services.gripper_state]
    missing = [s for s in must_srvs if not wait_for_service(s, cfg.startup_timeout_s)]
    if missing:
        rospy.logerr("[startup] missing services:\n  " + "\n  ".join(missing))
        raise SystemExit(1)

    rospy.loginfo("[startup] ready ✅")

    # build interfaces
    quest = Quest2Interface(debug=False)  # for haptics output + convenience
    robot = XArmRobot(auto_init=True, debug=False)

    teleop = QuestXArmTeleopSync(cfg=cfg, quest=quest, robot=robot)

    # Example hook for later dataset writing
    # def hook(sample: SyncedSample):
    #     pass
    # teleop.register_hook(hook)

    rospy.loginfo("[main] teleop sync running (ATS).")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        sys.exit(int(e.code))
    except Exception as e:
        try:
            rospy.logerr(f"Fatal: {e}")
        except Exception:
            print(f"Fatal: {e}")
        sys.exit(1)
