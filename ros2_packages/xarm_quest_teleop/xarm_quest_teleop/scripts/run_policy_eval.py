# xarm_quest_teleop/scripts/run_policy_eval.py
# usage:
#   ros2 run xarm_quest_teleop run_policy_eval
#   ros2 run xarm_quest_teleop run_policy_eval --ros-args -p config:=cleanup_table_d2_rollout.yaml
from __future__ import annotations

import os
import sys
import threading
from dataclasses import fields

from xarm_quest_teleop.ros import compat as rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "seeker-dev")))

from xarm_quest_teleop.io.camera import TwoRgbSync
from xarm_quest_teleop.io.process_manager import ManagedProcess, wait_for_topic, wait_for_service, SampleRing, ProcessSupervisor
from xarm_quest_teleop.configs.teleop_config import TeleopConfig
from xarm_quest_teleop.configs.eval_config import EvalConfig
from xarm_quest_teleop.configs.ros2_config import package_share_dir
from xarm_quest_teleop.configs.robot_config import ROBOT_TOPIC, XArmServices
from xarm_quest_teleop.robots.xarm import XArmRobot

from xarm_quest_teleop.policy.registry import create_policy
from xarm_quest_teleop.eval.eval_runner import EvalRunner


EVAL_CONFIG_ALIASES = {
    "profile": "eval_profile",
    "eval_profile": "eval_profile",
    "ckpt": "model_ckpt_path",
    "checkpoint": "model_ckpt_path",
    "model_ckpt": "model_ckpt_path",
    "model_ckpt_path": "model_ckpt_path",
    "task": "task_name",
    "task_name": "task_name",
    "name": "model_name",
    "model_name": "model_name",
    "policy": "policy_name",
    "policy_name": "policy_name",
    "policy_kwargs": "policy_kwargs",
    "replay_cache": "replay_cache_dir",
    "replay_cache_dir": "replay_cache_dir",
    "replay_episode": "replay_episode",
    "replay_start": "replay_start",
    "eval_name": "eval_name",
    "run_name": "eval_name",
    "condition": "eval_name",
    "calib": "calibration_path",
    "calibration": "calibration_path",
    "calibration_path": "calibration_path",
}


def _ros_param(name: str, default):
    if rospy.has_param(name):
        return rospy.get_param(name)
    return default


def _repo_root() -> str:
    return str(package_share_dir())


def _resolve_eval_config_path(path_value: str) -> str:
    raw_path = os.path.expanduser(str(path_value))
    if os.path.isabs(raw_path):
        return raw_path

    repo_root = _repo_root()
    candidates = [
        os.path.abspath(raw_path),
        os.path.join(repo_root, raw_path),
        os.path.join(repo_root, "config", "eval", raw_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[1]


def _load_eval_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "YAML eval configs require PyYAML. Install python3-yaml or pyyaml."
        ) from exc

    resolved = _resolve_eval_config_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Eval YAML config not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Eval YAML must contain a mapping: {resolved}")
    if isinstance(payload.get("eval"), dict):
        payload = payload["eval"]
    payload = dict(payload)
    payload["_resolved_path"] = resolved
    return payload


def _yaml_value(payload: dict, *keys: str):
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _apply_eval_yaml_values(cfg: EvalConfig, payload: dict) -> None:
    valid_fields = {field.name for field in fields(EvalConfig)}
    source = payload.get("_resolved_path", "<eval yaml>")
    cfg.eval_config_path = source

    for key, value in payload.items():
        if key.startswith("_") or key in {"eval", "profile", "eval_profile"}:
            continue
        attr = EVAL_CONFIG_ALIASES.get(str(key), str(key))
        if attr not in valid_fields:
            rospy.logwarn(f"[eval config] ignoring unknown key {key!r} in {source}")
            continue
        setattr(cfg, attr, value)


def _apply_eval_profile_defaults(cfg: EvalConfig) -> None:
    profile = str(cfg.eval_profile).strip().lower().replace("-", "_")
    profile_aliases = {
        "debug": "dry_run",
        "dryrun": "dry_run",
        "no_actuate": "dry_run",
        "real": "rollout",
        "actuate": "rollout",
        "run": "rollout",
    }
    profile = profile_aliases.get(profile, profile)
    cfg.eval_profile = profile

    if profile == "manual":
        return

    if profile == "dry_run":
        cfg.debug_no_actuate = True
        cfg.record = False
        cfg.model_name = "real_rvt2_dry_run"
        return

    if profile == "rollout":
        cfg.debug_no_actuate = False
        cfg.record = True
        cfg.model_name = "real_rvt2_policy"
        return

    raise ValueError(
        f"Unknown eval profile {cfg.eval_profile!r}. "
        "Expected one of: dry_run, rollout, manual."
    )


def _apply_ros_eval_params(cfg: EvalConfig) -> None:
    eval_yaml_path = _ros_param("~config", cfg.eval_config_path)
    yaml_payload = _load_eval_yaml(eval_yaml_path)
    yaml_profile = _yaml_value(yaml_payload, "profile", "eval_profile")
    if yaml_profile is not None:
        cfg.eval_profile = str(yaml_profile)
    _apply_eval_profile_defaults(cfg)
    _apply_eval_yaml_values(cfg, yaml_payload)
    cfg.enable_rgb_sync = True
    cfg.enable_full_sync = False


def _apply_launch_defaults(cfg: EvalConfig, teleop_cfg: TeleopConfig) -> None:
    cfg.launch.workdir = teleop_cfg.launch_workdir
    cfg.launch.pipe_output = teleop_cfg.pipe_launch_output
    cfg.launch.robot_cmd = list(teleop_cfg.ROBOT_LAUNCH_CMD)


def _apply_checkpoint_policy_contract(cfg: EvalConfig, net) -> None:
    ckpt_task_name = str(getattr(net, "task_name", "") or "")
    if ckpt_task_name and ckpt_task_name != str(cfg.task_name):
        rospy.logwarn(
            f"[eval startup] cfg.task_name={cfg.task_name} does not match "
            f"checkpoint task_name={ckpt_task_name}; using checkpoint value"
        )
        cfg.task_name = ckpt_task_name

    policy_obs_horizon = int(getattr(net, "n_obs_steps", cfg.obs_horizon))
    if int(cfg.obs_horizon) != policy_obs_horizon:
        rospy.logwarn(
            f"[eval startup] cfg.obs_horizon={cfg.obs_horizon} does not match "
            f"checkpoint n_obs_steps={policy_obs_horizon}; using checkpoint value"
        )
        cfg.obs_horizon = policy_obs_horizon

    cfg.pred_horizon = int(getattr(net, "n_action_steps", cfg.pred_horizon))
    cfg.exec_horizon = min(int(cfg.exec_horizon), max(1, int(cfg.pred_horizon) - 1))


def _log_real_eval_runtime_config(cfg: EvalConfig) -> None:
    rospy.loginfo(
        "[eval startup] real runtime config after checkpoint load: "
        f"config={cfg.eval_config_path} profile={cfg.eval_profile} "
        f"task={cfg.task_name} model_name={cfg.model_name} "
        f"eval_name={cfg.eval_name} seed={cfg.seed} "
        f"obs_horizon={cfg.obs_horizon} pred_horizon={cfg.pred_horizon} "
        f"exec_horizon={cfg.exec_horizon} rgb_cams={list(cfg.rgb_cams)} "
        f"control_hz={cfg.control_hz} interp_steps={cfg.interp_steps} "
        f"max_step_trans={cfg.max_step_trans} max_step_rot_rad={cfg.max_step_rot_rad} "
        f"workspace_min={list(cfg.workspace_min_xyz)} workspace_max={list(cfg.workspace_max_xyz)} "
        f"calibration_path={cfg.calibration_path} debug_no_actuate={cfg.debug_no_actuate} "
        f"live_viz={cfg.live_viz}"
    )


def _build_policy(cfg: EvalConfig):
    policy_name = str(getattr(cfg, "policy_name", "seeker") or "seeker")
    if policy_name == "cache_replay":
        kwargs = dict(getattr(cfg, "policy_kwargs", {}) or {})
        kwargs.setdefault("cache_dir", cfg.replay_cache_dir)
        kwargs.setdefault("episode", int(cfg.replay_episode))
        kwargs.setdefault("start", int(cfg.replay_start))
        kwargs.setdefault("horizon", int(cfg.pred_horizon))
        kwargs.setdefault("advance", int(cfg.exec_horizon))
        kwargs.setdefault("obs_horizon", int(cfg.obs_horizon))
        return create_policy(policy_name, **kwargs)

    kwargs = dict(getattr(cfg, "policy_kwargs", {}) or {})
    kwargs.setdefault("ckpt_path", cfg.model_ckpt_path)
    kwargs.setdefault("device", str(cfg.device))
    kwargs.setdefault("seed", int(cfg.seed))
    return create_policy(policy_name, **kwargs)


def main():
    sup = ProcessSupervisor()
    rospy.init_node("policy_eval", anonymous=False)

    teleop_cfg = TeleopConfig()
    services = XArmServices()

    cfg = EvalConfig()
    _apply_ros_eval_params(cfg)

    # launch commands
    _apply_launch_defaults(cfg, teleop_cfg)

    def shutdown():
        rospy.logwarn("[main] shutting down, stopping launched processes")
        sup.stop_all()

    rospy.on_shutdown(shutdown)

    # ---- autolaunch ----
    L = cfg.launch
    if L.enabled:
        if L.auto_launch_robot and L.robot_cmd:
            sup.start(ManagedProcess("xarm_driver", L.robot_cmd, L.workdir, L.pipe_output))

        if L.auto_launch_realsense:
            cmds = list(getattr(L, "realsense_rgb_launch_cmds", []))

            for i, cmd in enumerate(cmds):
                sup.start(ManagedProcess(f"realsense_{i}", cmd, L.workdir, L.pipe_output))

    # ---- wait robot ----
    rospy.loginfo("[eval startup] waiting for robot topic/services...")

    if not wait_for_topic(ROBOT_TOPIC, teleop_cfg.startup_timeout_s):
        rospy.logerr(f"[eval startup] missing robot topic: {ROBOT_TOPIC}")
        raise SystemExit(1)

    must_srvs = [
        services.set_mode, services.set_state, services.move_servo_cart,
        services.gripper_move, services.gripper_state
    ]
    missing = [s for s in must_srvs if not wait_for_service(s, teleop_cfg.startup_timeout_s)]
    if missing:
        rospy.logerr("[eval startup] missing services:\n  " + "\n  ".join(missing))
        raise SystemExit(1)

    # ---- wait cameras ----
    if cfg.enable_rgb_sync:
        for cam_name, spec in cfg.cam_sync.cameras_rgb.items():
            if not wait_for_topic(spec.rgb_topic, teleop_cfg.startup_timeout_s):
                rospy.logerr(f"[eval startup] missing rgb camera topic ({cam_name}): {spec.rgb_topic}")
                raise SystemExit(1)

    rospy.loginfo("[eval startup] ready ✅")

    # ---- robot + ring ----
    robot = XArmRobot(auto_init=True, debug=False)

    sample_ring = SampleRing(
        keep_s=float(cfg.robot_sync.keep_s),
        maxlen=int(cfg.robot_sync.queue_maxlen),
    )

    # Eval always uses robot state directly. Quest teleop is collection-only.
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

    # ---- policy + merged runner ----
    net = _build_policy(cfg)
    _apply_checkpoint_policy_contract(cfg, net)
    _log_real_eval_runtime_config(cfg)

    runner = EvalRunner(cfg=cfg, robot=robot, net=net, sample_ring=sample_ring, result_log_dir=cfg.result_log_dir)
    runner.start()

    # ---- sync wiring ----
    if cfg.enable_rgb_sync:
        cam_rgb_dict = {k: {"rgb_topic": v.rgb_topic} for k, v in cfg.cam_sync.cameras_rgb.items()}
        rgb2 = TwoRgbSync(
            cameras=cam_rgb_dict,
            slop_s=float(cfg.cam_sync.rgb_slop_s),
            queue_size=int(cfg.cam_sync.rgb_queue_size),
        )
        rgb2.on_set = runner.on_rgb_set

    rospy.loginfo("[eval] running. Keyboard: c(start), p(pause), r(reset), s(success), f(fail), q(quit)")
    rospy.spin()


if __name__ == "__main__":
    main()
