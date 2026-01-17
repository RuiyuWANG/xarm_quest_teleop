#!/usr/bin/env python3
# usage: python replay_traj_by_command.py 
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import subprocess
import signal
import rospy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.robots.xarm import XArmRobot

ROBOT_LAUNCH_CMD = [
    "roslaunch",
    "xarm_bringup",
    "xarm7_server.launch",
    "robot_ip:=192.168.1.241",
    "report_type:=dev",
    "add_gripper:=true",
]

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


def load_lowdim(npz_path: str):
    data = np.load(npz_path, allow_pickle=False)

    required = ["timestamp", "pose_targets6", "pose_targets_gripper"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}")

    t = np.asarray(data["timestamp"], dtype=np.float64)
    pose6 = np.asarray(data["pose_targets6"], dtype=np.float32)
    grip = np.asarray(data["pose_targets_gripper"], dtype=np.float32)

    assert pose6.ndim == 2 and pose6.shape[1] == 6
    assert t.shape[0] == pose6.shape[0] == grip.shape[0]

    if not np.all(np.isfinite(pose6)):
        raise ValueError("pose_targets6 contains NaNs")
    if not np.all(np.isfinite(grip)):
        raise ValueError("pose_targets_gripper contains NaNs")

    return t, pose6, grip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=str, help="Episode directory containing lowdim.npz")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0 = real time)")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (-1 = all)")
    parser.add_argument("--dry_run", action="store_true", help="Do not move robot, just print")
    args = parser.parse_args()

    npz_path = os.path.join(args.episode_dir, "lowdim.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)

    rospy.init_node("replay_lowdim_demo", anonymous=False)
    p = None 
    try:
        p = launch_robot_node()
        rospy.loginfo(f"[replay] loading {npz_path}")
        t, pose6, grip = load_lowdim(npz_path)

        robot = XArmRobot(auto_init=True, debug=False)

        N = len(t)
        start = max(0, args.start)
        end = N if args.max_steps < 0 else min(N, start + args.max_steps)

        # normalize time
        t0 = t[start]
        t_rel = (t[start:end] - t0) / max(args.speed, 1e-6)

        rospy.loginfo(f"[replay] steps {start} → {end-1}, speed={args.speed}")

        wall_t0 = time.time()
        
        # robot.home()

        for k in range(start, end):
            if rospy.is_shutdown():
                break

            i = k - start
            target_wall = wall_t0 + float(t_rel[i])

            # wait until scheduled time
            while not rospy.is_shutdown():
                dt = target_wall - time.time()
                if dt <= 0:
                    break
                time.sleep(min(0.005, dt))

            p = pose6[k].tolist()
            g = float(grip[k])

            if args.dry_run:
                rospy.loginfo(f"[replay] k={k} pose6={p} gripper={g}")
                continue

            # --- absolute pose w.r.t. base ---
            try:
                robot.move_servo_cart(pose6_mm_rpy=p)
            except Exception as e:
                rospy.logerr(f"[replay] move_to_pose failed at k={k}: {e}")
                break

            # --- gripper ---
            try:
                robot.move_gripper(g)
            except Exception as e:
                rospy.logwarn(f"[replay] move_gripper failed at k={k}: {e}")

        rospy.loginfo("[replay] done")
        
    finally:
        print("[test] shutting down bringup")
        shutdown_launch(p)


if __name__ == "__main__":
    main()
