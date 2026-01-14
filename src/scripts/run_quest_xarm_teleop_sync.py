#!/usr/bin/env python3
import os
import sys
from typing import Optional, List

import rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.io.quest2 import Quest2Interface
from src.io.process_manager import ManagedProcess, wait_for_topic, wait_for_service
from src.robots.xarm import XArmRobot
from src.configs.teleop_config import TeleopConfig
from src.configs.robot_config import ROBOT_TOPIC, XArmServices

from src.teleop.quest_xarm_teleop_sync import QuestXArmTeleopSync

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

    # start wrapper nodes
    stamp_quest_cmd = ["rosrun", cfg.scripts_package, "quest_stamped_node.py"]

    p = ManagedProcess("quest_stamp_node", stamp_quest_cmd, cfg.launch_workdir, cfg.pipe_launch_output)
    p.start()
    procs.append(p)

    # wait for stamped topics + xarm services
    rospy.loginfo("[startup] waiting for stamped topics/services...")

    if cfg.active_hand == "right":
        pose_t = cfg.right_pose_stamped_topic
        inputs_t = cfg.right_inputs_stamped_topic
    else:
        pose_t = cfg.left_pose_stamped_topic
        inputs_t = cfg.left_inputs_stamped_topic

    must_topics = [pose_t, inputs_t]
    for t in must_topics:
        if not wait_for_topic(t, cfg.startup_timeout_s):
            rospy.logerr(f"[startup] missing quest topic: {t}")
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
