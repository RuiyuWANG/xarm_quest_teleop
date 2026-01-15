import os
import time
import numpy as np
import rclpy
import json
from PIL import Image
from asset.robot_ros2 import XArmRobot
from scipy.spatial.transform import Rotation as R, Slerp
import palm.utils.transform_utils as TUtils
from pynput import keyboard
import threading
import copy
import palm.utils.real_exp_image_utils as img_utils


class DemoCollector:
    def __init__(self, robot, save_dir, num_demos, init_range, steps):
        print("[Collector] Initializing DemoCollector...")

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.init_range = init_range
        self.num_demos = num_demos
        self.steps = steps

        self.robot = robot

        # Control flags
        self.delete_requested = False
        self.quit_requested = False
        self.continue_trajectory = False
        self.waiting_for_continue = False
        self.demo_in_progress = False
        self.demo_step = 0
        self.current_demo_id = 0
        self.start_rot_rpy = None

        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    def keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == "c":
                    if not self.demo_in_progress:
                        print(f"[Collector] Starting demo {self.current_demo_id}...")
                        self.demo_in_progress = True
                    else:
                        print("[Collector] Continuing trajectory execution.")
                        self.continue_trajectory = True
                elif key.char == "d":
                    print("[Collector] Delete requested.")
                    self.delete_requested = True
                elif key.char == "q":
                    print("[Collector] Quit requested.")
                    self.quit_requested = True
            except AttributeError:
                pass

        keyboard.Listener(on_press=on_press).start()

    def run_demo_step(self):
        if not self.demo_in_progress:
            return

        # Step 0: Initialization
        if self.demo_step == 0:
            self.robot.home_robot()
            if not self.robot.has_states():
                print("[Collector] Waiting for robot states...")
                return
            self.obj_xy, init_pose, gripper_pos = self.robot.get_robot_init_state(
                0, self.init_range
            )
            self.start_rot_rpy = init_pose[3:]
            self.target_object_pose = self.get_next_target_pose(self.obj_xy)
            self.robot.move_to_init_state(init_pose, gripper_pos, wait=True)
            self.reach_steps = int((1 + np.random.uniform(-0.2, 0.2)) * self.steps)
            self.down_steps = 5 + np.random.randint(-1, 1)
            self.z_offset = 0.03 + np.random.uniform(-0.01, 0.01)
            traj = self.reach_then_down(
                self.robot.robot_state["eef_pose"],
                self.target_object_pose,
                z_offset=self.z_offset,
                reach_steps=self.reach_steps,
                down_steps=self.down_steps,
            )
            self.trajectory = traj
            self.traj_idx = 0
            self.demo_to_save = []
            print("[Collector] Press 'c' to start trajectory.")
            self.continue_trajectory = False
            self.demo_step += 1

        # Step 1: Wait for user to continue
        elif self.demo_step == 1:
            if not self.continue_trajectory:
                return
            # Execute trajectory step-by-step
            if self.traj_idx < len(self.trajectory):
                act = self.trajectory[self.traj_idx]
                self.demo_to_save.append(
                    self.format_data_to_save(
                        step=self.traj_idx,
                        robot_action=act,
                    )
                )
                self.robot.move_to_pose(
                    self.convert_trajectory(act), wait=True, speed=100.0, acc=500.0
                )
                self.traj_idx += 1
            else:
                self.demo_step += 1
                # gripper trajectory
                n = 5 + np.random.randint(-2, 2)
                gripper_trajectory = np.linspace(850, 330, num=n).astype(float).tolist()
                self.gripper_trajectory = gripper_trajectory[1:]  # Skip first position
                self.gripper_idx = 0

        # Step 2: Gripper closing
        elif self.demo_step == 2:
            if self.gripper_idx < len(self.gripper_trajectory):
                pos = self.gripper_trajectory[self.gripper_idx]
                self.demo_to_save.append(
                    self.format_data_to_save(
                        step=self.traj_idx,
                        gripper_action=pos,
                    )
                )
                self.robot.move_gripper(pos, wait=True)
                self.traj_idx += 1
                self.gripper_idx += 1
                return
            else:
                self.demo_to_save.append(
                    self.format_data_to_save(
                        step=self.traj_idx,
                        gripper_action=self.gripper_trajectory[-1],
                    )
                )
                self.traj_idx += 1
                self.demo_step += 1
                self.save_demo(self.current_demo_id, self.demo_to_save, subtask_name="subtask0")

        # Step 3: Move to place position
        elif self.demo_step == 3:
            pose_when_pick = self.robot.robot_state["eef_pose"]
            obj_xy, robot_xyz_rpy, gripper_pos = self.robot.get_robot_init_state(
                1, self.init_range
            )
            target_object_pose = self.get_next_target_pose(
                obj_xy, mode="place", past_pose=pose_when_pick
            )
            robot_xyz_rpy[3:] = self.start_rot_rpy  # Use the initial rotation from the first demo
            self.robot.move_to_init_state(robot_xyz_rpy, gripper_pos, wait=True)
            place_steps = int((1.1 + np.random.uniform(-0.2, 0.2)) * self.steps)
            assert place_steps > 0, "Place steps must be positive."
            #
            target_object_pose[:3, :3] = self.robot.robot_state["eef_pose"][
                :3, :3
            ]  # Keep the same orientation
            traj = self.reach_then_down(
                self.robot.robot_state["eef_pose"],
                target_object_pose,
                z_offset=0,
                reach_steps=place_steps,
            )
            self.trajectory = traj
            self.traj_idx = 0
            self.demo_to_save = []
            self.demo_step += 1

        # Step 4: Execute place trajectory
        elif self.demo_step == 4:
            if self.traj_idx < len(self.trajectory):
                act = self.trajectory[self.traj_idx]
                self.demo_to_save.append(
                    self.format_data_to_save(
                        step=self.traj_idx,
                        robot_action=act,
                    )
                )
                self.robot.move_to_pose(
                    self.convert_trajectory(act), wait=True, speed=100.0, acc=500.0
                )
                self.traj_idx += 1
            else:
                self.demo_step += 1
                n = 5 + np.random.randint(-2, 2)
                gripper_trajectory = np.linspace(330, 850, num=n).astype(float).tolist()
                self.gripper_trajectory = gripper_trajectory[1:]  # Skip first position
                self.gripper_idx = 0

        # Step 5: Gripper opening
        elif self.demo_step == 5:
            if self.gripper_idx < len(self.gripper_trajectory):
                pos = self.gripper_trajectory[self.gripper_idx]
                self.demo_to_save.append(
                    self.format_data_to_save(
                        step=self.traj_idx,
                        gripper_action=pos,
                    )
                )
                self.robot.move_gripper(pos, wait=True)
                self.traj_idx += 1
                self.gripper_idx += 1
                return
            else:
                self.demo_to_save.append(
                    self.format_data_to_save(
                        step=self.traj_idx,
                        gripper_action=self.gripper_trajectory[-1],
                    )
                )
                self.traj_idx += 1
                self.demo_step += 1
                self.save_demo(self.current_demo_id, self.demo_to_save, subtask_name="subtask1")

        # Step 6: Save demo and prepare for next
        elif self.demo_step == 6:
            print(f"[Collector] Demo {self.current_demo_id} completed and saved.")
            print("[Collector] 'c' to start/continue demo, 'd' to delete last demo, 'q' to quit.")
            self.demo_step = 0
            self.demo_in_progress = False
            self.current_demo_id += 1

    # ------------------------- Utility methods -------------------------

    def get_demo_id(self):
        demo_id = 0
        demo_path = os.path.join(self.save_dir, "subtask0")
        demo_path_1 = os.path.join(self.save_dir, "subtask1")
        if not os.path.exists(demo_path):
            os.makedirs(demo_path, exist_ok=True)
            os.makedirs(demo_path_1, exist_ok=True)
            return demo_id, 0
        existing_demos = [d for d in os.listdir(demo_path) if d.startswith("episode")]
        existing_demos_1 = [d for d in os.listdir(demo_path_1) if d.startswith("episode")]
        assert len(existing_demos) == len(
            existing_demos_1
        ), "Subtask directories must have the same number of demos."
        if existing_demos:
            existing_ids = [int(d.split("episode")[1]) for d in existing_demos]
            demo_id = max(existing_ids) + 1
        return demo_id, len(existing_demos)

    def linear_interpolate(self, start_pose, end_pose, steps):
        assert start_pose.shape == (4, 4) and end_pose.shape == (4, 4)
        t_start = start_pose[:3, 3]
        t_end = end_pose[:3, 3]
        t_vecs = np.linspace(t_start, t_end, steps)

        key_times = [0, 1]
        key_rots = R.from_matrix([start_pose[:3, :3], end_pose[:3, :3]])
        slerp = Slerp(key_times, key_rots)
        fractions = np.linspace(0, 1, steps)
        r_rots = slerp(fractions)

        out_poses = []
        for t, r in zip(t_vecs, r_rots):
            new_pose = np.eye(4)
            new_pose[:3, :3] = r.as_matrix()
            new_pose[:3, 3] = t
            out_poses.append(new_pose)
        return np.array(out_poses)

    def reach_then_down(self, start_pose, end_pose, z_offset, reach_steps, down_steps=None):
        if z_offset == 0:
            full_trajectory = self.linear_interpolate(start_pose, end_pose, steps=reach_steps)
        else:
            assert down_steps is not None, "down_steps must be provided if z_offset is not zero."
            mid_pose = end_pose.copy()
            mid_pose[2, 3] += z_offset
            traj_to_mid = self.linear_interpolate(start_pose, mid_pose, steps=reach_steps)
            traj_to_end = self.linear_interpolate(mid_pose, end_pose, steps=down_steps)
            full_trajectory = np.concatenate((traj_to_mid, traj_to_end), axis=0)
        return full_trajectory[1:]  # Skip the first pose to avoid redundancy

    def convert_trajectory(self, trajectory):
        assert isinstance(trajectory, np.ndarray) and trajectory.shape[-2:] == (4, 4)
        has_batch_dim = trajectory.ndim == 3
        if not has_batch_dim:
            trajectory = trajectory[None, ...]
        converted_traj = []
        for X in trajectory:
            xyz_rpy = TUtils.SE3_to_Eular(X)
            xyz_rpy[:3] = xyz_rpy[:3] * 1000.0  # Convert m to mm
            converted_traj.append(xyz_rpy.tolist())
        return converted_traj if has_batch_dim else converted_traj[0]

    def format_data_to_save(self, step, robot_action=None, gripper_action=None):
        def cp(d):
            return d.copy() if isinstance(d, np.ndarray) else copy.copy(d)

        gripper_qpos = cp(
            gripper_action if gripper_action is not None else self.robot.gripper_raw_pos
        )

        robot_action = cp(
            robot_action if robot_action is not None else self.robot.robot_state["eef_pose"]
        )

        data = {
            "step": int(step),
            "gripper_qpos": np.array([gripper_qpos]),
            "action": robot_action,
            "eef_pose": self.robot.robot_state["eef_pose"].copy(),
            "rgb_frame": self.robot.rgb_frame["front_rgb_raw"].copy(),
            "raw_eef_pose": self.robot.raw_robot_state.copy(),
        }
        return data

    def save_demo(self, demo_id, subtask_data, subtask_name):
        subtask_path = os.path.join(self.save_dir, subtask_name, f"episode{demo_id}")
        im_dir = os.path.join(subtask_path, "images")
        os.makedirs(im_dir, exist_ok=True)
        json_path = os.path.join(subtask_path, "low_dim.json")

        low_dim_json = {}
        for sample in subtask_data:
            step = sample["step"]
            low_dim_json[step] = {
                "eef_pose": sample["eef_pose"].tolist(),
                "raw_eef_pose": sample["raw_eef_pose"].tolist(),
                "gripper_qpos": sample["gripper_qpos"].tolist(),
                "action": sample["action"].tolist(),
            }
            Image.fromarray(sample["rgb_frame"]).save(os.path.join(im_dir, f"{step}.png"))
        with open(json_path, "w") as f:
            json.dump(low_dim_json, f, indent=4)

    def delete_demo(self, demo_id):
        for subtask in ["subtask0", "subtask1"]:
            subtask_path = os.path.join(self.save_dir, subtask, f"episode{demo_id}")
            if os.path.exists(subtask_path):
                try:
                    json_file = os.path.join(subtask_path, "low_dim.json")
                    if os.path.exists(json_file):
                        os.remove(json_file)
                    im_dir = os.path.join(subtask_path, "images")
                    if os.path.exists(im_dir):
                        for img_file in os.listdir(im_dir):
                            os.remove(os.path.join(im_dir, img_file))
                        os.rmdir(im_dir)
                    os.rmdir(subtask_path)
                    print(f"[Collector] Deleted demo {demo_id} from {subtask}.")
                except Exception as e:
                    print(f"[Error] Failed to delete demo {demo_id} from {subtask}: {e}")
            else:
                print(f"[Error] Demo {demo_id} does not exist in {subtask}.")

    def get_next_target_pose(self, target_xy, mode="pick", past_pose=None, object_height=0.035):
        assert mode in ["pick", "place"], "Mode must be 'pick' or 'place'."
        X = np.eye(4)
        curr_pose = self.robot.robot_state["eef_pose"]
        if mode == "pick":
            curr_R = curr_pose[:3, :3]
            obj_rotation = img_utils.sample_ee_xy_rotation(curr_R, angle_range_deg=(-45, 45))
            new_z = object_height + np.random.uniform(-0.015, 0.01)
            X[:3, :3] = obj_rotation
            X[:2, 3] = target_xy[:2]
            X[2, 3] = new_z
        elif mode == "place":
            assert past_pose is not None, "past_pose must be provided for 'place' mode."
            new_z = past_pose[2, 3] + np.random.uniform(0.0, 0.01) + object_height
            X = past_pose.copy()
            X[:2, 3] = target_xy[:2]
            X[2, 3] = new_z
        self.robot.target_object_pose = X
        return X


# ------------------------- Main Runner -------------------------
def main(args):
    rclpy.init()
    # Initialize robot (Node) and collector (plain Python class)
    robot = XArmRobot(
        img_size=(200, 200),
        crop_kwargs={"tcp_crop_size": (240, 240), "tcp_height_offset": 60},
    )

    collector = DemoCollector(
        save_dir=args.save_dir,
        num_demos=args.num_demos,
        init_range=args.init_range,
        steps=args.steps,
        robot=robot,
    )

    collector.current_demo_id, collector.num_demos = collector.get_demo_id()

    print("[Collector] 'c' to start/continue demo, 'd' to delete last demo, 'q' to quit.")
    while rclpy.ok() and not collector.quit_requested:
        try:
            rclpy.spin_once(robot, timeout_sec=0.1)
            collector.run_demo_step()
            if not collector.demo_in_progress:
                _, curr_num_demos = collector.get_demo_id()
                if collector.delete_requested:
                    collector.delete_requested = False
                    collector.delete_demo(collector.current_demo_id - 1)
                if curr_num_demos <= args.num_demos:
                    collector.demo_step = 0
                else:
                    print("[Collector] All demos completed.")

        except Exception as e:
            print(f"[Collector] Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the demo collector for robot demonstrations."
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_demos", type=int, required=True)
    parser.add_argument("--init_range", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()

    main(args)
