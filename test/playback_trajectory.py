import os
import time
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from palm.utils.transform_utils import Eular_to_SE3

ROBOT_IP = "192.168.1.244"
SAVE_ROOT = "data/real_stack_replay"
TRAJ_DIR = "data/real_stack"
RECORD_FREQ = 10  # Hz

os.makedirs(SAVE_ROOT, exist_ok=True)

def get_eef_pose(arm):
    _, pose = arm.get_position(is_radian=False)
    if _ != 0:
        raise RuntimeError("Failed to get robot pose.")
    return pose  # [x, y, z, roll, pitch, yaw]

def get_gripper(arm):
    return arm.get_gripper_position()

def setup_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def record_demo(arm, camera, demo_name):
    # Replay trajectory
    print(f"==> Replaying {demo_name}")
    ret = arm.playback_trajectory(times=1, filename=os.path.join(TRAJ_DIR, demo_name), wait=False)
    if ret != 0:
        print(f"[!] Failed to play trajectory: {ret}")
        return

    # Create save path
    demo_save_path = os.path.join(SAVE_ROOT, demo_name.replace('.traj', ''))
    os.makedirs(os.path.join(demo_save_path, "images"), exist_ok=True)
    state_log = []

    t_start = time.time()
    frame_id = 0

    print("==> Recording data during playback...")

    while True:
        # Poll robot state
        try:
            pose = get_eef_pose(arm)
            gripper = get_gripper(arm)
        except:
            break  # probably disconnected or motion stopped

        # Poll image
        frames = camera.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())

        # Save image
        img_name = f"{frame_id:04d}.png"
        img_path = os.path.join(demo_save_path, "images", img_name)
        cv2.imwrite(img_path, color_image)

        # Save state
        state_log.append({
            "timestamp": time.time() - t_start,
            "pose": pose,
            "gripper": gripper,
            "image": img_name,
        })

        frame_id += 1
        time.sleep(1 / RECORD_FREQ)

        # Check if playback is done
        state = arm.get_state()
        if state in [4]:  # stopped or idle
            print("==> Playback finished.")
            break

    # Save metadata
    with open(os.path.join(demo_save_path, "states.json"), "w") as f:
        json.dump(state_log, f, indent=2)

    print(f"[✓] Saved {len(state_log)} frames to {demo_save_path}")

def main():
    # Initialize robot
    arm = XArmAPI(ROBOT_IP, is_radian=True)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1)

    # Initialize camera
    camera = setup_camera()

    # Get .traj files
    traj_files = sorted([f for f in os.listdir(TRAJ_DIR) if f.endswith('.traj')])

    print(f"Found {len(traj_files)} trajectories.")
    for traj in traj_files:
        record_demo(arm, camera, traj)
        time.sleep(1)

    camera.stop()
    arm.disconnect()
    print("All done.")

if __name__ == "__main__":
    main()