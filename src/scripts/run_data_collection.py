#!/usr/bin/env python3
import os
import sys
from typing import List

import rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.io.quest2 import Quest2Interface
from src.io.camera import CameraSync
from src.io.process_manager import ManagedProcess, wait_for_topic, wait_for_service

from src.utils.config_utils import load_dataset_yaml
from src.configs.teleop_config import TeleopConfig
from src.configs.robot_config import ROBOT_TOPIC, XArmServices
from src.configs.collector_config import CollectorConfig

from src.robots.xarm import XArmRobot
from src.teleop.quest_xarm_teleop_sync import QuestXArmTeleopSync
from src.data_collection.colloter import TeleopDataCollector


def resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(pkg_root, p)


def main():
    rospy.init_node("teleop_data_collection", anonymous=False)

    teleop_cfg = TeleopConfig()
    services = XArmServices()

    # dataset yaml
    # TODO: make this configurable
    dataset_yaml = rospy.get_param("~dataset_yaml", "config/test.yaml")
    dataset_yaml = resolve_path(dataset_yaml)
    ds = load_dataset_yaml(dataset_yaml)

    # collector config
    collector_cfg = CollectorConfig()

    # Fill launch commands from TeleopConfig
    collector_cfg.launch.workdir = teleop_cfg.launch_workdir
    collector_cfg.launch.pipe_output = teleop_cfg.pipe_launch_output
    collector_cfg.launch.quest_cmd = list(teleop_cfg.QUEST_LAUNCH_CMD)
    collector_cfg.launch.robot_cmd = list(teleop_cfg.ROBOT_LAUNCH_CMD)
    collector_cfg.launch.stamp_cmd = ["rosrun", teleop_cfg.scripts_package, "quest_stamped_node.py"]

    # Share timing policy: camera matching window >= ATS slop and >= one servo tick
    collector_cfg.cam_sync.match_window_s = max(
        float(teleop_cfg.sync_slop_s),
        1.0 / float(teleop_cfg.servo_rate_hz),
    )

    procs: List[ManagedProcess] = []

    def shutdown():
        rospy.logwarn("[main] shutting down, stopping launched processes")
        for p in reversed(procs):
            try:
                p.stop()
            except Exception:
                pass

    rospy.on_shutdown(shutdown)

    # Auto-launch quest + robot + realsense + stamp node
    L = collector_cfg.launch
    if L.enabled:
        if L.auto_launch_quest and L.quest_cmd:
            p = ManagedProcess("quest_ros_tcp_endpoint", L.quest_cmd, L.workdir, L.pipe_output)
            p.start()
            procs.append(p)

        if L.auto_launch_robot and L.robot_cmd:
            p = ManagedProcess("xarm_bringup", L.robot_cmd, L.workdir, L.pipe_output)
            p.start()
            procs.append(p)

        if L.auto_launch_realsense:
            for i, cmd in enumerate(L.realsense_cmds):
                p = ManagedProcess(f"realsense_{i}", cmd, L.workdir, L.pipe_output)
                p.start()
                procs.append(p)

        if L.auto_launch_stamp_node and L.stamp_cmd:
            p = ManagedProcess("quest_stamp_node", L.stamp_cmd, L.workdir, L.pipe_output)
            p.start()
            procs.append(p)

    # Wait readiness: quest topics, robot topic/services
    rospy.loginfo("[startup] waiting for topics/services...")

    if teleop_cfg.active_hand == "right":
        pose_t = teleop_cfg.right_pose_stamped_topic
        inputs_t = teleop_cfg.right_inputs_stamped_topic
    else:
        pose_t = teleop_cfg.left_pose_stamped_topic
        inputs_t = teleop_cfg.left_inputs_stamped_topic

    for t in [pose_t, inputs_t, ROBOT_TOPIC]:
        if not wait_for_topic(t, teleop_cfg.startup_timeout_s):
            rospy.logerr(f"[startup] missing topic: {t}")
            raise SystemExit(1)

    must_srvs = [
        services.set_mode, services.set_state, services.velo_move_line_timed,
        services.gripper_move, services.gripper_state
    ]
    missing = [s for s in must_srvs if not wait_for_service(s, teleop_cfg.startup_timeout_s)]
    if missing:
        rospy.logerr("[startup] missing services:\n  " + "\n  ".join(missing))
        raise SystemExit(1)

    # Wait for camera RGB topics (cloud may appear slightly later; we don't block on it)
    for cam_name, spec in collector_cfg.cam_sync.cameras.items():
        if not wait_for_topic(spec.rgb_topic, teleop_cfg.startup_timeout_s):
            rospy.logerr(f"[startup] missing camera rgb topic: {spec.rgb_topic}")
            raise SystemExit(1)

    rospy.loginfo("[startup] ready ✅")

    # Build interfaces
    quest = Quest2Interface(debug=False)
    robot = XArmRobot(auto_init=True, debug=False)
    teleop = QuestXArmTeleopSync(cfg=teleop_cfg, quest=quest, robot=robot)

    # CameraSync + Collector
    cam_dict = {
        k: {"rgb_topic": v.rgb_topic, "cloud_topic": v.cloud_topic}
        for k, v in collector_cfg.cam_sync.cameras.items()
    }
    cam_sync = CameraSync(
        cam_dict,
        keep_s=collector_cfg.cam_sync.keep_s,
        match_window_s=collector_cfg.cam_sync.match_window_s,
        queue_size=collector_cfg.cam_sync.queue_size,
    )

    collector = TeleopDataCollector(ds, collector_cfg, robot=robot)

    # Hook: very fast (attach msgs + enqueue). Heavy saving is in collector worker thread.
    def collector_hook(sample):
        sample.cameras = cam_sync.nearest(sample.stamp_sync)  # cloud chosen nearest to rgb stamp
        collector.enqueue(sample)

    teleop.register_hook(collector_hook)

    rospy.loginfo("[main] data collection running. Press 'c' start, 's' save, ''d' delete, 'q' quit.")
    rospy.spin()


if __name__ == "__main__":
    main()