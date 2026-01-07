import os
import time
import numpy as np
from pynput import keyboard

import cv2
import threading
from img_utils import *
import json
import palm.utils.transform_utils as TUtils

import rospy
from asset.robot import XArmRobot


class DemoCollector:
    def __init__(self, save_dir, num_demos, init_range, freq):
        self.save_dir = save_dir
        self.init_range = init_range
        self.num_demos = num_demos
        self.freq = freq

        self.gripper_open = True
        self.id = 0
        self.image = []
        self.last_z = 0
        self.target_obj_pose = np.eye(4)

        self.image_lock = threading.Lock()
        self.stop_recording = threading.Event()

        os.makedirs(self.save_dir, exist_ok=True)
        subtask0_dir = os.path.join(self.save_dir, "subtask0")
        subtask1_dir = os.path.join(self.save_dir, "subtask1")

        if os.path.exists(subtask0_dir) and os.path.exists(subtask1_dir):
            episodes_0 = [
                d for d in os.listdir(subtask0_dir) if os.path.isdir(os.path.join(subtask0_dir, d))
            ]
            episodes_1 = [
                d for d in os.listdir(subtask1_dir) if os.path.isdir(os.path.join(subtask1_dir, d))
            ]

            assert len(episodes_0) == len(episodes_1), (
                f"Mismatch: subtask0 has {len(episodes_0)} folders, "
                f"subtask1 has {len(episodes_1)} folders."
            )
            self.current_demo_idx = len(episodes_0)
        else:
            self.current_demo_idx = 0

        # Initialize robot
        self.robot = XArmRobot(
            img_size=(200, 200),
            crop_kwargs={
                "tcp_crop_size": (240, 240),
                "tcp_height_offset": 60,
            },
        )
        # Move to initial position
        self.robot.home_robot()

    def keyboard_listener(self):
        print("Keyboard listener started (press b/s/c/q)...")

        def on_press(key):
            try:
                if key.char == "b":
                    target_object = self.robot.move_robot_to_init(0, self.init_range)

                    target_obj_pose = np.eye(4)
                    R_curr = self.robot.robot_state["eef_pose"][:3, :3]
                    obj_rotation = sample_ee_xy_rotation(R_curr, angle_range_deg=(-45, 45))
                    new_z = 0.035 + np.random.uniform(-0.015, 0.01)
                    target_obj_pose[:3, :3] = obj_rotation
                    target_obj_pose[:2, 3] = target_object[:2]
                    target_obj_pose[2, 3] = new_z
                    self.target_obj_pose = target_obj_pose
                    self.robot.target_object = target_obj_pose

                    print(
                        f"[INFO] Begin demo collection {self.current_demo_idx + 1}/{self.num_demos}, subtask {self.id}"
                    )

                elif key.char == "s":
                    self.mixed_trajectory_replay_and_record()
                    print(f"[INFO] Demo {self.current_demo_idx + 1} saved")

                    self.current_demo_idx += 1
                    if self.current_demo_idx >= self.num_demos:
                        print("[INFO] All demos collected.")
                        self.robot.home_robot()
                        return

                    self.id = 0
                    self.last_z = 0
                    self.gripper_open = True
                    self.image = []
                    self.target_obj_pose = np.eye(4)
                    self.robot.home_robot()

                elif key.char == "c":
                    print("[INFO] Demo canceled.")

                    self.id = 0
                    self.last_z = 0
                    self.gripper_open = True
                    self.image = []
                    self.target_obj_pose = np.eye(4)
                    self.robot.home_robot()

                elif key.char == "q":
                    print("[INFO] Quit requested.")
                    self.stop_recording.set()
                    self.robot.home_robot()
                    return False

            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def meta_policy(self, target_position=None, task="stack_two"):
        if task == "stack_two":
            if self.id == 0:  # go down and grasp
                robot_state_6d = self.robot.raw_robot_state

                # go x y
                inter_z = 110 + np.random.uniform(-0.015, 0.015) * 1000
                self.robot.move_to_pose(
                    [
                        self.target_obj_pose[0, 3] * 1000,
                        self.target_obj_pose[1, 3] * 1000,
                        inter_z,
                        robot_state_6d[3],
                        robot_state_6d[4],
                        robot_state_6d[5],
                    ]
                )

                # rotation, go down, close gripper
                new_rot = TUtils.SE3_to_Eular(self.target_obj_pose)
                new_z = self.target_obj_pose[2, 3] * 1000
                self.last_z = new_z
                self.robot.move_to_pose(
                    [
                        self.target_obj_pose[0, 3] * 1000,
                        self.target_obj_pose[1, 3] * 1000,
                        new_z,
                        new_rot[3],
                        new_rot[4],
                        new_rot[5],
                    ]
                )
                rospy.sleep(0.1)
                self.robot.move_gripper(330)
                # HACK: wait for gripper to close
                rospy.sleep(0.1)
                # update status
                self.id = 1
                self.gripper_open = False

            elif self.id == 1:  # go down and place the cube
                robot_state_6d = self.robot.raw_robot_state

                new_z = (self.last_z / 1000 + 0.035 + np.random.uniform(0, 0.01)) * 1000
                if target_position is not None:
                    new_eef_pose = [target_position[0], target_position[1]]
                else:
                    new_eef_pose = [robot_state_6d[0], robot_state_6d[1]]
                self.robot.move_to_pose(
                    [
                        new_eef_pose[0],
                        new_eef_pose[1],
                        new_z,
                        robot_state_6d[3],
                        robot_state_6d[4],
                        robot_state_6d[5],
                    ]
                )
                self.robot.move_gripper(850)
                self.gripper_open = True
        else:
            raise NotImplementedError

    def mixed_trajectory_replay_and_record(self):
        # Create subtask directories
        subtask0_path = os.path.join(self.save_dir, "subtask0")
        subtask1_path = os.path.join(self.save_dir, "subtask1")
        os.makedirs(subtask0_path, exist_ok=True)
        os.makedirs(subtask1_path, exist_ok=True)

        traj0_path = os.path.join(subtask0_path, f"episode{self.current_demo_idx}")
        traj1_path = os.path.join(subtask1_path, f"episode{self.current_demo_idx}")
        os.makedirs(traj0_path, exist_ok=True)
        os.makedirs(traj1_path, exist_ok=True)

        # Record and execute subtask0 (meta policy for grasping)
        self.stop_recording.clear()
        recording_thread0 = threading.Thread(target=self.record_demo, args=(traj0_path,))
        recording_thread0.start()

        # Execute meta policy for grasping
        self.meta_policy(task="stack_two")

        # Lift after grasping
        robot_state_6d = self.robot.raw_robot_state
        robot_state_6d[2] = robot_state_6d[2] + 50
        self.robot.move_to_pose(
            [
                robot_state_6d[0],
                robot_state_6d[1],
                robot_state_6d[2],
                robot_state_6d[3],
                robot_state_6d[4],
                robot_state_6d[5],
            ]
        )

        # Stop recording for subtask0
        self.stop_recording.set()
        recording_thread0.join()

        # Lift after grasping
        robot_state_6d = self.robot.raw_robot_state
        robot_state_6d[2] = 160
        self.robot.move_to_pose(
            [
                robot_state_6d[0],
                robot_state_6d[1],
                robot_state_6d[2],
                robot_state_6d[3],
                robot_state_6d[4],
                robot_state_6d[5],
            ]
        )

        # Move to placing position (transition - not recorded)
        target_object = self.robot.move_robot_to_init(1, self.init_range)
        time.sleep(0.1)

        # Record and execute subtask1 (meta policy for placing)
        self.stop_recording.clear()
        recording_thread1 = threading.Thread(target=self.record_demo, args=(traj1_path,))
        recording_thread1.start()

        # Execute meta policy for placing
        self.meta_policy(target_position=target_object * 1000, task="stack_two")

        # Lift after placing (transition - not recorded)
        robot_state_6d = self.robot.raw_robot_state
        robot_state_6d[2] = robot_state_6d[2] + 50
        self.robot.move_to_pose(
            [
                robot_state_6d[0],
                robot_state_6d[1],
                robot_state_6d[2],
                robot_state_6d[3],
                robot_state_6d[4],
                robot_state_6d[5],
            ]
        )

        # Stop recording for subtask1
        self.stop_recording.set()
        recording_thread1.join()

    def record_demo(self, save_path):
        os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
        low_dim_data = {}
        count = 0
        last_time = time.time()

        while not self.stop_recording.is_set():
            current_time = time.time()
            if current_time - last_time < 1.0 / self.freq:
                time.sleep(0.001)
                continue
            last_time = current_time

            try:
                # Get low dim data
                ee_pose = self.robot.robot_state["eef_pose"]
                raw_eef_pose = self.robot.raw_robot_state
                gripper_qpos = self.robot.gripper_qpos
                # process img
                img = self.robot.rgb_frame["front_rgb"]

                # Save low_dim data
                low_dim_data[count] = {
                    "ee_pose": ee_pose.tolist(),
                    "gripper_qpos": float(gripper_qpos),
                    "raw_eef_pose": raw_eef_pose.tolist(),
                }

                # Save image
                img_path = os.path.join(save_path, "images", f"{count}.png")
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                count += 1

            except Exception as e:
                print(f"Recording error: {e}")

        # Save low_dim_data to JSON
        with open(os.path.join(save_path, "low_dim.json"), "w") as f:
            json.dump(low_dim_data, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="../data/real_stack_sync_mix_5")
    parser.add_argument("--num_demos", type=int, default=155)
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--init_range", type=float, default=0.08)
    args = parser.parse_args()

    rospy.init_node("palm_eval_ros", anonymous=True)

    demo_collector = DemoCollector(
        save_dir=args.save_dir,
        num_demos=args.num_demos,
        init_range=args.init_range,
        freq=args.freq,
    )

    # Start keyboard listener
    demo_collector.keyboard_listener()
