# usage: python run_data_collection.py [~dataset_json:=test.json]
import os
import sys
from typing import List
import copy
import rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.io.quest2 import Quest2Interface
from src.io.camera import CamRgbDepthSync, TwoRgbSync
from src.io.process_manager import ManagedProcess, ProcessSupervisor, wait_for_topic, wait_for_service, SampleRing

from src.utils.config_utils import load_dataset_json
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
    return os.path.join(pkg_root, "config", p)


def main():
    sup = ProcessSupervisor()
    rospy.init_node("teleop_data_collection", anonymous=False)

    teleop_cfg = TeleopConfig()
    services = XArmServices()

    dataset_json = rospy.get_param("~dataset_json", "three_piece_toy_d2.json")
    dataset_json = resolve_path(dataset_json)
    ds = load_dataset_json(dataset_json)

    collector_cfg = CollectorConfig()

    # Fill launch commands from TeleopConfig
    collector_cfg.launch.workdir = teleop_cfg.launch_workdir
    collector_cfg.launch.pipe_output = teleop_cfg.pipe_launch_output
    collector_cfg.launch.quest_cmd = list(teleop_cfg.QUEST_LAUNCH_CMD)
    collector_cfg.launch.robot_cmd = list(teleop_cfg.ROBOT_LAUNCH_CMD)
    collector_cfg.launch.stamp_cmd = ["rosrun", teleop_cfg.scripts_package, "quest_stamped_node.py"]

    def shutdown():
        rospy.logwarn("[main] shutting down, stopping launched processes")
        sup.stop_all()

    rospy.on_shutdown(shutdown)

    # Auto-launch quest + robot + realsense + stamp node
    L = collector_cfg.launch
    if L.enabled:
        if L.auto_launch_quest and L.quest_cmd:
            sup.start(ManagedProcess("quest_ros_tcp_endpoint", L.quest_cmd, L.workdir, L.pipe_output))

        if L.auto_launch_robot and L.robot_cmd:
            sup.start(ManagedProcess("xarm_bringup", L.robot_cmd, L.workdir, L.pipe_output))

        if L.auto_launch_realsense:
            cmds = L.realsense_all_launch_cmds if collector_cfg.enable_full_sync else L.realsense_light_launch_cmds
            for i, cmd in enumerate(cmds):
                sup.start(ManagedProcess(f"realsense_{i}", cmd, L.workdir, L.pipe_output))

        if L.auto_launch_stamp_node and L.stamp_cmd:
            sup.start(ManagedProcess("quest_stamp_node", L.stamp_cmd, L.workdir, L.pipe_output))

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

    # Wait for camera topics based on enabled syncs
    assert not (not collector_cfg.enable_full_sync and not collector_cfg.enable_light_sync), \
        "At least one camera sync must be enabled"

    if collector_cfg.enable_full_sync:
        for cam_name, spec in collector_cfg.cam_sync.cameras_all.items():
            if not wait_for_topic(spec.rgb_topic, teleop_cfg.startup_timeout_s):
                rospy.logerr(f"[startup] missing full camera rgb topic ({cam_name}): {spec.rgb_topic}")
                raise SystemExit(1)
            if not wait_for_topic(spec.depth_topic, teleop_cfg.startup_timeout_s):
                rospy.logerr(f"[startup] missing full camera depth topic ({cam_name}): {spec.depth_topic}")
                raise SystemExit(1)

    if collector_cfg.enable_light_sync:
        for cam_name, spec in collector_cfg.cam_sync.cameras_light.items():
            if not wait_for_topic(spec.rgb_topic, teleop_cfg.startup_timeout_s):
                rospy.logerr(f"[startup] missing light camera rgb topic ({cam_name}): {spec.rgb_topic}")
                raise SystemExit(1)

    rospy.loginfo("[startup] ready ✅")

    # Build interfaces
    quest = Quest2Interface(debug=False)
    robot = XArmRobot(auto_init=True, debug=False)
    teleop = QuestXArmTeleopSync(cfg=teleop_cfg, quest=quest, robot=robot)
    collector = TeleopDataCollector(ds, collector_cfg, robot=robot)

    # ---------------- Teleop ring ----------------
    sample_ring = SampleRing(
        keep_s=float(collector_cfg.robot_sync.keep_s),
        maxlen=int(collector_cfg.robot_sync.queue_maxlen),
    )

    def teleop_hook(sample):
        sample_ring.push(sample)

    teleop.register_hook(teleop_hook)

    # ---------------- FULL sync: 3 cams RGB+Depth ----------------
    if collector_cfg.enable_full_sync:
        cam_all_dict = {
            k: {"rgb_topic": v.rgb_topic, "depth_topic": v.depth_topic}
            for k, v in collector_cfg.cam_sync.cameras_all.items()
        }

        cam_sync3 = CamRgbDepthSync(
            cameras=cam_all_dict,
            pair_slop_s=float(collector_cfg.cam_sync.pair_slop_s),
            tri_slop_s=float(collector_cfg.cam_sync.tri_slop_s),
            pair_buf_len=int(collector_cfg.cam_sync.pair_buf_len),
            keep_s=float(collector_cfg.cam_sync.keep_s),
            max_wait_s=float(collector_cfg.cam_sync.max_wait_s),
            ref_cam=str(collector_cfg.cam_sync.ref_camera),
            sub_queue_size=int(collector_cfg.cam_sync.sub_queue_size),
        )

        def on_full_set(t_cam, cams_set):
            t_cam = float(t_cam)

            s = sample_ring.nearest(t_cam, float(collector_cfg.robot_sync.robot_match_window_s))
            if s is None:
                rospy.logwarn_throttle(
                    2.0,
                    f"[CamRobotSync] REF t={t_cam:.3f}: cameras has no paired robot sample within tri_slop={collector_cfg.robot_sync.robot_match_window_s}s"
                )
                return
            if not bool(getattr(s, "allow_control", False)):
                return

            s_full = copy.copy(s)
            s_full.cameras = cams_set
            s_full.stamp_sync = t_cam
            collector.enqueue_all(s_full)

        cam_sync3.on_set = on_full_set

    # ---------------- LIGHT sync: 2 cams RGB only ----------------
    if collector_cfg.enable_light_sync:
        cam_light_dict = {
            k: {"rgb_topic": v.rgb_topic}
            for k, v in collector_cfg.cam_sync.cameras_light.items()
        }

        cam_rgb_sync2 = TwoRgbSync(
            cameras=cam_light_dict,
            slop_s=float(collector_cfg.cam_sync.rgb_slop_s),
            queue_size=int(collector_cfg.cam_sync.rgb_queue_size),
        )

        def on_rgb2_set(t_cam, imgs):
            t_cam = float(t_cam)

            s = sample_ring.nearest(t_cam, float(collector_cfg.robot_sync.robot_match_window_s))
            if s is None or not bool(getattr(s, "allow_control", False)):
                return

            s_light = copy.copy(s)
            s_light.cameras = imgs
            s_light.stamp_sync = t_cam
            collector.enqueue_light(s_light)

        cam_rgb_sync2.on_set = on_rgb2_set

    rospy.loginfo("[main] data collection running. Press 'c' start, 's' save, 'd' delete, 'q' quit.")
    rospy.spin()


if __name__ == "__main__":
    main()
