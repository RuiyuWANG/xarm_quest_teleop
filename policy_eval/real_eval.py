import time
import copy
import threading
import argparse
import os
import numpy as np
import cv2
import json
import torch
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# robot and camera
from xarm.wrapper import XArmAPI
import pyrealsense2 as rs

from palm.models.bc_mlp import BCMLP
from palm.utils.net_utils import parse_network_configs
from palm.utils.config_utils import load_config, dict_to_namespace, load_config_to_namespace
import palm.utils.image_utils as PalmUtils
import palm.utils.transform_utils as TUtils
from img_utils import *

# === home pose in joint space [j1, j2, j3, j4, j5, j6, j7] in angle ===
HOME = [0, -45, 0, 45, 0, 90, 0]
# === Initial position [x, y, z, roll, pitch, yaw, gripper_open] ===
# INIT_POSE = [155, 16, 170, 3.14193, 0, 0, 500]
ROBOT_IP = "192.168.1.244"
ROBOT_CAM_IP = "192.168.1.243"

# camera matrices
INTRINSICS_FILE = "charuco_intrinsics.npz"
HAND_EYE_FILE = "handeye_result.npz"
# EYE_HAND_FILE = "eye_hand_result.npz"

handeye_data = np.load(HAND_EYE_FILE)
intr_data = np.load(INTRINSICS_FILE)
# eyehand_data = np.load(EYE_HAND_FILE)

X_C = handeye_data["base_T_cam"]
K = intr_data["K"]
# T = eyehand_data["base_T_cam"]
T = np.eye(4)

"""
    For the XArm gripper, 800 is the wideset open, 0 is close 
"""


def get_new_cam_matrix_in_a(rad, X_C):
    X_M = np.eye(4)

    M_X_C = np.linalg.inv(X_M) @ X_C

    # Rotate the camera around the movement center
    delta_X = np.eye(4)
    delta_X[:3, :3] = R.from_rotvec(rad * np.array([0, 0, 1])).as_matrix()
    M_X_C = delta_X @ M_X_C

    X_C_prime = X_M @ M_X_C

    return X_C_prime


def get_new_cam_matrix_in_b(curr_ee, X_C, X_C_prime, T):
    ee_prime = curr_ee @ T @ np.linalg.inv(X_C) @ X_C_prime @ np.linalg.inv(T)
    return ee_prime


class PolicyRunner:
    def __init__(
        self,
        ckpt_path,
        horizon,
        device="cuda",
        freq=10,
        num_rollouts=30,
        obs_step=1,
        result_log_path=None,
        is_radian=False,
        use_meta=True,
        is_palm=False,
        cam_angle=0.0,
    ):
        assert result_log_path is not None, "Result log path must be provided."
        self.result_log_path = result_log_path
        self.exp_list = list(range(1, num_rollouts + 1))  # [1, 2, 3, ...]
        self.is_radian = is_radian
        self.current_exp_idx = 1
        self.obs_steps = obs_step
        self.last_episode_status = None
        self.success_count = 0
        self.X_C = X_C
        self.cam_angle = cam_angle
        self.is_palm = is_palm

        # Thread synchronization
        self.obs_lock = threading.Lock()
        self.action_available = threading.Condition()
        self.new_obs_event = threading.Event()
        self.should_stop = threading.Event()
        self.allow_action_execution = threading.Event()
        self.restart_requested = False

        # Parameters
        self.img_size = (200, 200)
        self.freq = freq

        # Observation buffers (store multiple steps)
        self.obs_buffer = {
            "front_rgb": [],
            "eef_pose": [],
            "eef_6d_xyz": [],
            "gripper_qpos": [],
            "tick": 0,
            "front_rgb_raw": [],
            "eef_6d_in_cam_z": [],
            "front_rgb_overlay_tcp_crop": [],
        }
        self.inferred_actions = {}
        self.action_mode = "delta"

        # Initialize robot and camera stream
        self.robot = XArmAPI(ROBOT_IP)
        self.camera = setup_camera()

        self.last_gripper_state = 1

        # Load policy from checkpoint
        dataset_config_dir = os.path.dirname(ckpt_path)
        dataset_config_path = os.path.join(dataset_config_dir, "config.json")
        cfg = parse_network_configs(dataset_config_path)
        policy = BCMLP(cfg)
        policy.load_state_dict(torch.load(ckpt_path)["net_params"], strict=True)
        self.device = torch.device(device)
        self.policy = policy.to(self.device).eval()

        # palm kwargs
        if is_palm:
            with open(dataset_config_path, "r") as f:
                args = json.load(f)
            args = dict_to_namespace(args)
            dataset_path = args.dataset.dataset_path
            dataset_dir = os.path.dirname(dataset_path)
            json_file_paths = [f for f in os.listdir(dataset_dir) if f.endswith("config.json")]
            assert len(json_file_paths) == 1, f"Multiple JSON files found in {dataset_dir}"
            dataset_cfg = load_config_to_namespace(
                os.path.join(dataset_dir, json_file_paths[0]), full_path=True
            )
            conversion_args = dataset_cfg.conversion

            self.crop_kwargs = {
                "tcp_crop_size": conversion_args.tcp_crop_size,
                "K": K,
                "tcp_height_offset": conversion_args.tcp_height_offset,
                "ops": conversion_args.ops.rgb,
                "resize_size": (conversion_args.img_size[1], conversion_args.img_size[0]),
            }
        else:
            self.crop_kwargs = None

        # Set up meta policy
        self.use_meta = use_meta
        self.id = 0

        # Move robot to home pose and record initial EE pose
        self.steps = 0
        self.horizon = horizon
        self.prev_action_tick = -1
        self.prev_gripper_open = True

        self.robot.motion_enable(enable=True)
        self.robot.set_gripper_enable(enable=True)
        self.robot.set_mode(0)  # position control mode
        self.robot.set_gripper_mode(0)
        self.robot.set_state(0)
        self.robot.set_collision_tool_model(1)  # xArm gripper
        print("============= Using position control mode =============")
        self.home()

    def get_subtask_id(self):
        if self.get_gripper_pose() < 400 and self.id >= 0:
            subtask_id = 1
        else:
            subtask_id = self.id
        return subtask_id

    def is_local(self):
        if self.use_meta:
            is_local = self.id == self.get_subtask_id()
        else:
            is_local = True
        return is_local

    def record_last_episode_result(self):
        if self.current_exp_idx >= len(self.exp_list):
            return
        exp_id = self.exp_list[self.current_exp_idx]
        line = f"{exp_id}: {self.last_episode_status}\n"
        # Ensure the directory exists (if path includes folders)
        os.makedirs(os.path.dirname(self.result_log_path), exist_ok=True)

        with open(self.result_log_path, "a") as f:  # 'a' creates the file if it doesn't exist
            f.write(line)

        self.current_exp_idx += 1

    def keyboard_listener(self):
        print("Keyboard listener started (press s/f/r)...")

        def on_press(key):
            try:
                if key.char == "s":
                    print("Key 's' pressed: marking as SUCCESS")
                    self.last_episode_status = "success"
                    self.record_last_episode_result()
                    self.success_count += 1
                    self.restart_requested = True

                elif key.char == "f":
                    print("Key 'f' pressed: marking as FAIL")
                    self.last_episode_status = "fail"
                    self.record_last_episode_result()
                    self.restart_requested = True

                elif key.char == "r":
                    print("Key 'r' pressed: marking as RESET")
                    self.last_episode_status = "reset"
                    self.restart_requested = True

                if self.current_exp_idx > 0:
                    success_rate = (self.success_count / self.current_exp_idx) * 100
                    print(
                        f"Success rate: {self.success_count}/{self.current_exp_idx} = {success_rate:.1f}%"
                    )

            except AttributeError:
                pass  # Ignore special keys

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def check_action_vals(self, x, y, z, roll, pitch, yaw):
        """
        min-max range:
        x: 155 to 500
        y: -320 to 380
        z: 10 to 400
        roll: 2.443 to -2.443     (in rad, 140 to -140°, going through -180/180)
        pitch: -0.5236 to  0.5236 (in rad, -30 to 30°)
        yaw: -0.7854 to  0.7854 (in rad, -45 to 45°)
        """
        # clip values using min max values
        x_safe = max(155, min(x, 500))
        y_safe = max(-320, min(y, 380))
        z_safe = max(10, min(z, 400))

        # ensure that roll is wrapped around
        if roll > np.pi:
            roll = -np.pi + (roll - np.pi)
        elif roll < -np.pi:
            roll = np.pi + (roll + np.pi)

        # check roll
        if roll >= 0 and roll < 2.443:
            roll = 2.443
        elif roll < 0 and roll > -2.443:
            roll = -2.443

        pitch_safe = max(-0.5236, min(pitch, 0.5236))
        yaw_safe = max(-0.7854, min(yaw, 0.7854))

        return x_safe, y_safe, z_safe, roll, pitch_safe, yaw_safe

    def home(self):
        """Send the robot to the predefined home configuration."""
        self.robot.set_servo_angle(servo_id=8, angle=HOME, wait=True, is_radian=False)

        if self.prev_gripper_open is True:
            self.robot.set_gripper_position(850, speed=5000, wait=True)

        # with self.obs_lock:
        #     self.obs_buffer = {
        #         "front_rgb": [],
        #         "eef_pose": [],
        #         "eef_6d_xyz": [],
        #         "eef_6d_in_cam_z": [],
        #         "front_rgb_overlay_tcp_crop": [],
        #         "gripper_qpos": [],
        #         "tick": 0,
        #         "front_rgb_raw": [],
        #     }
        time.sleep(1)

    def get_eef_pose(self):
        code, pose = self.robot.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError("Failed to get robot pose.")
        pose = np.array(pose)
        pose[:3] = pose[:3] / 1000  # Convert mm to m
        X = TUtils.Eular_to_SE3(pose)  # Convert to SE(3)
        return X

    def get_gripper_pose(self):
        """Get current gripper qpos as a (1,) array"""
        code, gripper_qpos = self.robot.get_gripper_position()
        if code != 0:
            raise RuntimeError("Failed to get gripper qpose")
        return np.array(gripper_qpos)

    def format_image(self, image: np.ndarray) -> np.ndarray:
        """Convert to RGB, Resize, and convert image to CHW format."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        resized = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255
        return resized

    def get_cropped_image(
        self,
        image: np.ndarray,
        crop_kwargs: dict,
    ) -> np.ndarray:
        ops = crop_kwargs["ops"]
        if "overlay_tcp_crop" in ops:
            ee_poses = crop_kwargs["eef_poses"]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image_overlay = PalmUtils.overlay_poses(images=image, poses=ee_poses, K=crop_kwargs["K"], X_C=self.X_C, axis_length=0.08)
            coords = PalmUtils.project_points(K=crop_kwargs["K"], X_C=self.X_C, poses=ee_poses)
            image_crop = PalmUtils.crop_at_coords(
                images=image,
                coords=coords,
                crop_size=crop_kwargs["tcp_crop_size"],
                height_offset=crop_kwargs["tcp_height_offset"],
            )

            image_resize = cv2.resize(
                image_crop, crop_kwargs["resize_size"], interpolation=cv2.INTER_LINEAR
            )

        return image_resize, image

    def image_callback(self):
        """
        Synchronized callback for both camera images. Processes and stores
        images and updates observations.
        """

        frames = self.camera.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("No color frame received.")

        agentview_image = np.asanyarray(color_frame.get_data())

        h, w = agentview_image.shape[:2]
        margin = (w - h) // 2
        if margin >= 0:
            centered_image = agentview_image[:, margin : margin + h]
        else:
            centered_image = agentview_image

        formatted_agentview_image = self.format_image(centered_image)
        crop_kwargs = self.crop_kwargs

        with self.obs_lock:
            self.obs_buffer["front_rgb"] = [formatted_agentview_image]
            self.obs_buffer["front_rgb_raw"] = [agentview_image]
            self.format_eef_pose()
            if crop_kwargs is not None:
                crop_kwargs["eef_poses"] = self.obs_buffer["eef_pose"][-1]
                tcp_cropped_image, image_overlay = self.get_cropped_image(
                    agentview_image, crop_kwargs
                )
                tcp_cropped_image_resized = (
                    np.transpose(tcp_cropped_image, (2, 0, 1)).astype(np.float32) / 255
                )
                self.obs_buffer["front_rgb_overlay_tcp_crop"] = [tcp_cropped_image_resized]
        # # Optional visualization
        viz = cv2.resize(image_overlay, self.img_size, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Observation", cv2.cvtColor(tcp_cropped_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        self.new_obs_event.set()

    def image_stream_loop(self):
        """Continuously grab frames and update observations."""
        while not self.should_stop.is_set():
            try:
                self.image_callback()
                time.sleep(1 / self.freq)
            except Exception as e:
                print(f"[image_stream_loop] Error: {e} ?")
                time.sleep(0.1)  # Prevent tight loop on error

    def format_eef_pose(self):
        """Convert EE pose and delta pose to 6D format, store in buffer."""
        curr_pose = self.get_eef_pose()
        curr_gripper = self.get_gripper_pose()
        eef_6d_xyz = TUtils.SE3_to_6D_xyz(curr_pose)

        C_X_H = np.linalg.inv(self.X_C) @ curr_pose
        X_Hybrid = C_X_H.copy()
        X_Hybrid[:3, 3] = curr_pose[:3, 3]
        eef_6d_in_cam_z = TUtils.SE3_to_6D_z(X_Hybrid)

        self.obs_buffer["eef_6d_xyz"] = [eef_6d_xyz]
        self.obs_buffer["eef_pose"] = [curr_pose]
        self.obs_buffer["gripper_qpos"] = [curr_gripper]
        self.obs_buffer["eef_6d_in_cam_z"] = [eef_6d_in_cam_z]
        self.obs_buffer["tick"] += 1

    def move_robot_to_init(self, ran_range=0.00):
        # Wait until an image is available
        while not self.should_stop.is_set():
            self.new_obs_event.wait(timeout=1.0)
            self.new_obs_event.clear()

            with self.obs_lock:
                if len(self.obs_buffer["front_rgb_raw"]) > 0:
                    img = self.obs_buffer["front_rgb_raw"][-1]
                    break
            print("No image in buffer yet...")

        else:
            raise RuntimeError("Stopped before image was available.")
        object_mask = get_object_mask(img, id=self.id, task="stack_two")
        object_position_in_pixel = get_object_position_from_mask(object_mask)
        table_height = 0.035
        object_position = cam_pt_to_table_pt(object_position_in_pixel, table_height)

        init_pose = np.array(
            [object_position[0], object_position[1], 160, 178.485864, 0.001318, 1.393949]
        )
        # TODO: hard coded
        x, y = init_pose[:2]
        if ran_range != 0.00:
            offset = np.random.uniform(-ran_range, ran_range, size=2)
            x = x + offset[0]
            y = y + offset[1]
        init_pose = [x * 1000, y * 1000, init_pose[2], init_pose[3], init_pose[4], init_pose[5]]

        code, joint_angles = self.robot.get_inverse_kinematics(
            init_pose, input_is_radian=False, return_is_radian=False
        )
        if code != 0:
            raise RuntimeError(f"Failed to get inverse kinematics: {code}")
        self.robot.set_servo_angle(servo_id=8, angle=joint_angles, wait=True, is_radian=False)

        gripper_pos = 850 if self.prev_gripper_open else 300
        self.robot.set_gripper_position(gripper_pos, wait=True)
        return object_position * 1000

    def align_eef_rotation_to_camera(self, X_Cs=(None, None)):
        X_H_prime = self.get_eef_pose()
        # align camera
        if all(x is not None for x in X_Cs):
            # rotate ee w.r.t. camera
            xc, X_C_prime = X_Cs
            C_X_H_prime = np.linalg.inv(xc) @ X_H_prime
            X_H_double_prime = X_C_prime @ C_X_H_prime
            X_H_prime[:3, :3] = X_H_double_prime[:3, :3]
        else:
            return

        X_H_prime_to_execute = TUtils.SE3_to_Eular(X_H_prime).flatten()
        self.robot.set_position(
            yaw=X_H_prime_to_execute[2],
            pitch=X_H_prime_to_execute[1],
            roll=X_H_prime_to_execute[0],
            wait=True,
            is_radian=True,
        )
        self.X_C = X_C_prime

    def predict_action(self, obs) -> torch.Tensor:
        if self.is_palm:
            obs_formatted = {
                "obs": {
                    "rgb": {
                        "front_rgb_overlay_tcp_crop": obs["front_rgb_overlay_tcp_crop"],
                    },
                    "low_dim": {
                        "eef_6d_in_cam_z": obs["eef_6d_in_cam_z"],
                    },
                    # "low_dim": {
                    #     "eef_6d_xyz": obs["eef_6d_xyz"],
                    # }
                }
            }
        else:
            obs_formatted = {
                "obs": {
                    "rgb": {
                        "front_rgb": obs["front_rgb"],
                    },
                    "low_dim": {
                        "eef_6d_xyz": obs["eef_6d_xyz"],
                    },
                }
            }
        with torch.no_grad():
            act_pred = self.policy(obs_formatted)
            # HARDCODED: this action key is hardcoded
            act_pred_denorm = self.policy.obs_normalizer.denormalize_actions(
                act_pred, keys=["delta_6d_xyz", "gripper_action"]
            )
        return act_pred_denorm

    def pass_obs_check(self):
        """Check if we have enough observations for inference."""
        with self.obs_lock:
            return all(
                len(buf) >= self.obs_steps for k, buf in self.obs_buffer.items() if k != "tick"
            )

    def infer_action_loop(self):
        """Loop that waits for new obs and runs policy inference."""
        while not self.should_stop.is_set():
            # Wait for new observation
            self.new_obs_event.wait()
            self.new_obs_event.clear()

            # Skip if we don't have enough observations
            # if not self.pass_obs_check():
            #     continue
            try:
                with self.obs_lock:
                    obs_data = copy.deepcopy(self.obs_buffer)

                obs_data.pop("gripper_qpos")
                obs_data.pop("front_rgb_raw")
                if self.is_palm:
                    # obs_data.pop("eef_6d_xyz")
                    obs_data.pop("front_rgb")
                else:
                    # obs_data.pop("eef_6d_in_cam_z")
                    obs_data.pop("front_rgb_overlay_tcp_crop")

                # Process observation data
                obs = {}
                for key, buffer in obs_data.items():
                    if key == "tick":
                        continue
                    # Take the last obs_steps
                    obs[key] = np.array(buffer[-self.obs_steps :], dtype=np.float32)
                    # Add batch dimension
                    # obs[key] = np.expand_dims(obs[key], axis=0)

                # Extract EE pose and gripper state
                eef_pose = obs.pop("eef_pose")

                # Run inference
                obs_tensor = {
                    k: torch.from_numpy(v).float().to(self.device) for k, v in obs.items()
                }
                action = self.predict_action(obs_tensor).cpu().numpy()

                gripper_action = 1.0 if action[:, -1] > 0.5 else 0.0
                action = TUtils.SE3_from_6D_xyz(action[:, :-1])

                if self.action_mode == "delta":
                    action = eef_pose @ action
                elif self.action_mode == "absolute":
                    action = action
                elif self.action_mode == "relative":
                    raise NotImplementedError("Relative action mode not implemented.")
                else:
                    print(f"Unknown action mode: {self.action_mode}")
                    continue

                # Store inferred action
                with self.action_available:
                    tick = obs_data["tick"]
                    self.inferred_actions[tick] = (action, gripper_action)
                    # Remove old actions to prevent memory buildup
                    min_tick = tick - 5  # Keep last 5 actions
                    for t in list(self.inferred_actions.keys()):
                        if t < min_tick:
                            del self.inferred_actions[t]
                    self.action_available.notify_all()

            except Exception as e:
                print(f"[infer_action_loop] Error: {e}")

    def get_most_recent_action(self):
        """Get the most recent action from the inferred actions."""
        with self.action_available:
            if not self.inferred_actions:
                return None, None, None
            most_recent_tick = max(self.inferred_actions.keys())
            action, gripper_action = self.inferred_actions[most_recent_tick]
            return action, gripper_action, most_recent_tick

    def action_execution_loop(self):
        """Loop that executes inferred actions on the robot."""
        print("Starting action execution loop")

        while not self.should_stop.is_set() and self.allow_action_execution.is_set():
            if self.current_exp_idx >= len(self.exp_list):
                print("All experiments completed, shutting down.")
                return

            if self.restart_requested:
                print("Restart requested: rehoming and resetting state.")
                self.restart_requested = False
                self.prev_gripper_open = True
                # self.home()
                self.id = 0
                self.steps = 0
                self.prev_action_tick = -1
                if self.cam_angle != 0.0:
                    self.align_eef_rotation_to_camera(X_Cs=(X_C, self.X_C))
                if runner.use_meta:
                    # time.sleep(3)
                    self.home()
                    runner.move_robot_to_init()
                    print("Moved robot to initial pose.")
                continue

            # Wait for new action if needed
            if self.is_local():
                eef_act, gripper_act, act_tick = self.get_most_recent_action()
                if act_tick is None or act_tick == self.prev_action_tick:
                    with self.action_available:
                        self.action_available.wait(timeout=0.1)
                    continue

                self.prev_action_tick = act_tick
                print(f"Executing action at step {self.steps}/{self.horizon}...")

                # Execute action
                # Handle gripper state change
                if self.last_gripper_state != gripper_act:
                    if gripper_act < 0.5:  # Close gripper
                        self.robot.set_gripper_position(330, wait=True)
                        self.prev_gripper_open = False
                    else:  # Open gripper
                        self.robot.set_gripper_position(850, wait=True)
                        self.prev_gripper_open = True
                    self.last_gripper_state = gripper_act

                # Convert to Euler angles and execute
                eef_target = TUtils.SE3_to_Eular(eef_act).flatten()
                eef_target[:3] = eef_target[:3] * 1000
                x, y, z, roll, pitch, yaw = eef_target
                x_safe, y_safe, z_safe, roll_safe, pitch_safe, yaw_safe = self.check_action_vals(
                    x, y, z, roll, pitch, yaw
                )

                # Execute robot movement
                self.robot.set_position(
                    x=x_safe,
                    y=y_safe,
                    z=z_safe,
                    roll=roll_safe,
                    pitch=pitch_safe,
                    yaw=yaw_safe,
                    wait=True,
                    is_radian=True,
                )

                self.steps += 1
                time.sleep(1 / self.freq)
            else:
                self.robot.set_position(z=160, wait=True, is_radian=False)
                self.id = self.get_subtask_id()
                target_object = self.move_robot_to_init()
                time.sleep(1)
                print("Move robot to init position...")

            # Check horizon
            # if self.steps >= self.horizon:
            #     print("Reached horizon, going home...")
            #     self.home()
            #     self.record_last_episode_result()
            #     self.steps = 0
            #     self.prev_action_tick = -1
            #     self.last_gripper_state = 1
            #     time.sleep(1)
            #     continue


if __name__ == "__main__":
    # === Parse CLI arguments ===
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="Number of steps to run the policy for",
    )

    parser.add_argument(
        "--result_log_path",
        type=str,
        default="./real_eval_exps/stack_baseline_in_domain.txt",
    )

    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=30,
        help="Total number of experiments to run",
    )

    parser.add_argument(
        "--cam_angle",
        type=float,
        default=0.0,
        help="Camera angle to rotate in radians",
    )

    parser.add_argument(
        "-p",
        "--palm",
        action="store_true",
        help="running PALM",
    )

    args = parser.parse_args()

    # === Create policy runner ===
    runner = PolicyRunner(
        horizon=args.horizon,
        ckpt_path=args.ckpt_path,
        result_log_path=args.result_log_path,
        num_rollouts=args.num_rollouts,
        is_palm=args.palm,
        cam_angle=args.cam_angle,
    )

    # Start keyboard listener in a separate thread
    keyboard_thread = threading.Thread(target=runner.keyboard_listener, daemon=True)
    keyboard_thread.start()

    # Start only image stream thread first
    image_thread = threading.Thread(target=runner.image_stream_loop, daemon=True)
    image_thread.start()

    # rotate camera and robot wrt to the new camera pose
    if args.cam_angle != 0.0:
        robot_b = XArmAPI(ROBOT_CAM_IP)
        X_C_p = get_new_cam_matrix_in_a(args.cam_angle, X_C)
        code, pose = robot_b.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError("Failed to get robot pose.")
        pose = np.array(pose)
        pose[:3] = pose[:3] / 1000  # Convert mm to m
        curr_ee = TUtils.Eular_to_SE3(pose)  # Convert to SE(3)

        eef_b_prime = get_new_cam_matrix_in_b(curr_ee, X_C, X_C_p, T)
        eef_to_execute = TUtils.SE3_to_Eular(eef_b_prime).flatten()
        eef_to_execute[:3] = eef_to_execute[:3] * 1000
        robot_b.set_position(
            x=eef_to_execute[0],
            y=eef_to_execute[1],
            z=eef_to_execute[2],
            roll=eef_to_execute[3],
            pitch=eef_to_execute[4],
            yaw=eef_to_execute[5],
            wait=True,
            is_radian=True,
        )
        time.sleep(1)

        runner.align_eef_rotation_to_camera(X_Cs=(X_C, X_C_p))
        time.sleep(1)

    if runner.use_meta:
        runner.move_robot_to_init()
        print("Moved robot to initial pose.")
        runner.allow_action_execution.set()

    # Now start the rest of the threads
    thread_targets = [
        runner.infer_action_loop,
        runner.action_execution_loop,
    ]
    thread_targets = [threading.Thread(target=fn, daemon=True) for fn in thread_targets]
    for t in thread_targets:
        t.start()

    threads = [image_thread] + thread_targets

    try:
        while any(t.is_alive() for t in threads) and not runner.should_stop.is_set():
            time.sleep(1)
            for i, t in enumerate(threads):
                if not t.is_alive():
                    print(f"Thread {t.name or f'Thread-{i}'} died, restarting...")
                    new_t = threading.Thread(target=thread_targets[i])
                    new_t.daemon = True
                    new_t.start()
                    threads[i] = new_t

    except KeyboardInterrupt:
        print("Shutting down...")
        runner.should_stop.set()
        runner.camera.stop()
        runner.robot.disconnect()
        cv2.destroyAllWindows()
