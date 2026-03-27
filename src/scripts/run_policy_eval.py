# src/scripts/run_policy_eval.py
# usage: python src/scripts/run_policy_eval.py [~model_ckpt:=/path/to.ckpt] [~launch_quest:=false]
from __future__ import annotations

import os
import sys
import time
import threading
from typing import List

import rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.io.camera import TwoRgbSync, CamRgbDepthSync
from src.io.process_manager import ManagedProcess, wait_for_topic, wait_for_service, SampleRing, ProcessSupervisor
from src.configs.teleop_config import TeleopConfig
from src.configs.eval_config import EvalConfig
from src.configs.robot_config import ROBOT_TOPIC, XArmServices
from src.robots.xarm import XArmRobot

from src.teleop.quest_xarm_teleop_sync import QuestXArmTeleopSync
from src.io.quest2 import Quest2Interface

from src.policy.seeker_policy import SeekerPolicy
from src.eval.eval_runner import EvalRunner


def main():
    sup = ProcessSupervisor()
    rospy.init_node("policy_eval", anonymous=False)

    teleop_cfg = TeleopConfig()
    services = XArmServices()

    cfg = EvalConfig()
    cfg.model_ckpt_path = rospy.get_param("~model_ckpt", cfg.model_ckpt_path)
    cfg.launch_quest = bool(rospy.get_param("~launch_quest", cfg.launch_quest))
    cfg.seed = int(rospy.get_param("~seed", cfg.seed))
    cfg.result_log_dir = rospy.get_param("~result_log_dir", cfg.result_log_dir)

    # launch commands
    cfg.launch.workdir = teleop_cfg.launch_workdir
    cfg.launch.pipe_output = teleop_cfg.pipe_launch_output
    cfg.launch.robot_cmd = list(teleop_cfg.ROBOT_LAUNCH_CMD)
    cfg.launch.stamp_cmd = ["rosrun", teleop_cfg.scripts_package, "quest_stamped_node.py"]
    if cfg.launch_quest:
        cfg.launch.quest_cmd = list(teleop_cfg.QUEST_LAUNCH_CMD)

    def shutdown():
        rospy.logwarn("[main] shutting down, stopping launched processes")
        sup.stop_all()

    rospy.on_shutdown(shutdown)

    # ---- autolaunch ----
    L = cfg.launch
    if L.enabled:
        if cfg.launch_quest and L.auto_launch_quest and L.quest_cmd:
            sup.start(ManagedProcess("quest_ros_tcp_endpoint", L.quest_cmd, L.workdir, L.pipe_output))

        if L.auto_launch_robot and L.robot_cmd:
            sup.start(ManagedProcess("xarm_bringup", L.robot_cmd, L.workdir, L.pipe_output))

        if L.auto_launch_realsense:
            if hasattr(L, "realsense_all_launch_cmds"):
                cmds = list(L.realsense_all_launch_cmds)
            else:
                cmds = list(getattr(L, "realsense_cmds", []))

            for i, cmd in enumerate(cmds):
                sup.start(ManagedProcess(f"realsense_{i}", cmd, L.workdir, L.pipe_output))

        if cfg.launch_quest and L.auto_launch_stamp_node and L.stamp_cmd:
            sup.start(ManagedProcess("quest_stamp_node", L.stamp_cmd, L.workdir, L.pipe_output))

    # ---- wait robot ----
    rospy.loginfo("[eval startup] waiting for robot topic/services...")

    if not wait_for_topic(ROBOT_TOPIC, teleop_cfg.startup_timeout_s):
        rospy.logerr(f"[eval startup] missing robot topic: {ROBOT_TOPIC}")
        raise SystemExit(1)

    must_srvs = [
        services.set_mode, services.set_state, services.velo_move_line_timed,
        services.gripper_move, services.gripper_state
    ]
    missing = [s for s in must_srvs if not wait_for_service(s, teleop_cfg.startup_timeout_s)]
    if missing:
        rospy.logerr("[eval startup] missing services:\n  " + "\n  ".join(missing))
        raise SystemExit(1)

    # ---- wait cameras ----
    if cfg.enable_light_sync:
        for cam_name, spec in cfg.cam_sync.cameras_light.items():
            if not wait_for_topic(spec.rgb_topic, teleop_cfg.startup_timeout_s):
                rospy.logerr(f"[eval startup] missing light camera rgb topic ({cam_name}): {spec.rgb_topic}")
                raise SystemExit(1)

    if cfg.enable_full_sync:
        for cam_name, spec in cfg.cam_sync.cameras_all.items():
            if not wait_for_topic(spec.rgb_topic, teleop_cfg.startup_timeout_s):
                rospy.logerr(f"[eval startup] missing full camera rgb topic ({cam_name}): {spec.rgb_topic}")
                raise SystemExit(1)
            if not wait_for_topic(spec.depth_topic, teleop_cfg.startup_timeout_s):
                rospy.logerr(f"[eval startup] missing full camera depth topic ({cam_name}): {spec.depth_topic}")
                raise SystemExit(1)

    rospy.loginfo("[eval startup] ready ✅")

    # ---- robot + ring ----
    robot = XArmRobot(auto_init=True, debug=False)

    sample_ring = SampleRing(
        keep_s=float(cfg.robot_sync.keep_s),
        maxlen=int(cfg.robot_sync.queue_maxlen),
    )

    if cfg.launch_quest:
        quest = Quest2Interface(debug=False)
        teleop = QuestXArmTeleopSync(cfg=teleop_cfg, quest=quest, robot=robot)

        def teleop_hook(sample):
            sample_ring.push(sample)

        teleop.register_hook(teleop_hook)
    else:
        # no quest: push minimal samples from robot state
        def robot_ring_thread():
            r = rospy.Rate(90.0)
            while not rospy.is_shutdown():
                st = robot.get_state()

                class _S:
                    pass

                s = _S()
                s.stamp_sync = rospy.Time.now().to_sec()
                s.robot_state = st
                sample_ring.push(s)
                r.sleep()

        threading.Thread(target=robot_ring_thread, daemon=True).start()

    # ---- net + merged runner ----
    net = SeekerPolicy(
        ckpt_path=cfg.model_ckpt_path,
        device=str(cfg.device),
        seed=cfg.seed,
    )

    runner = EvalRunner(cfg=cfg, robot=robot, net=net, sample_ring=sample_ring, result_log_dir=cfg.result_log_dir)
    runner.start()

    # ---- sync wiring ----
    if cfg.enable_light_sync:
        cam_light_dict = {k: {"rgb_topic": v.rgb_topic} for k, v in cfg.cam_sync.cameras_light.items()}
        rgb2 = TwoRgbSync(
            cameras=cam_light_dict,
            slop_s=float(cfg.cam_sync.rgb_slop_s),
            queue_size=int(cfg.cam_sync.rgb_queue_size),
        )
        rgb2.on_set = runner.on_light_rgb_set

    if cfg.enable_full_sync:
        cam_all_dict = {k: {"rgb_topic": v.rgb_topic, "depth_topic": v.depth_topic} for k, v in cfg.cam_sync.cameras_all.items()}
        ref_cam = list(cam_all_dict.keys())[0]
        full = CamRgbDepthSync(
            cameras=cam_all_dict,
            pair_slop_s=float(cfg.cam_sync.pair_slop_s),
            tri_slop_s=float(cfg.cam_sync.tri_slop_s),
            pair_buf_len=int(cfg.cam_sync.pair_buf_len),
            keep_s=float(cfg.cam_sync.keep_s),
            max_wait_s=float(cfg.cam_sync.max_wait_s),
            ref_cam=ref_cam,
            sub_queue_size=int(cfg.cam_sync.sub_queue_size),
        )
        full.on_set = runner.on_full_rgbd_set

    rospy.loginfo("[eval] running. Keyboard: c(start), p(pause), r(reset), s(success), f(fail), q(quit)")
    rospy.spin()


if __name__ == "__main__":
    main()
