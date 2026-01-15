import numpy as np
import sys
import os
import time
import rospy

import torch
from asset.robot import XArmRobot

from palm.models.bc_mlp import BCMLP
from palm.utils.net_utils import parse_network_configs
import palm.utils.transform_utils as TUtils

from pynput import keyboard
import threading


class NetWrapper:
    def __init__(self, model_path):
        self.net = self.load_model(model_path)
        assert torch.cuda.is_available() or torch.backends.mps.is_available(), "No GPU available"
        self.net.to("cuda").eval()
        self.device = "cuda"

    def infer_action(self, obs_dict):
        normed_act = self.net(self.obs_dict_to_torch(obs_dict))
        # denormalize the action
        act = (
            self.net.obs_normalizer.denormalize_actions(
                normed_act, keys=["delta_6d_xyz", "gripper_action"]
            )
            .detach()
            .cpu()
            .numpy()
        )

        gripper_act = 1.0 if act[:, -1] > 0.5 else 0.0
        bot_act = TUtils.SE3_from_6D_xyz(act[:, :-1])
        return bot_act, gripper_act

    def numpy_image_to_tensor(self, numpy_image):
        assert isinstance(numpy_image, np.ndarray), "Input must be a numpy array"
        assert numpy_image.ndim == 3 and numpy_image.shape[2] == 3
        assert numpy_image.dtype == np.uint8, "Input image must be of type uint8"
        tensor_image = torch.from_numpy(numpy_image).permute(2, 0, 1).float() / 255.0
        tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension
        return tensor_image

    def obs_dict_to_torch(self, obs_dict):
        return {
            "obs": {
                "rgb": {
                    key: torch.from_numpy(obs_dict[key]).float().to(self.device)
                    for key in obs_dict
                    if "rgb" in key
                },
                "low_dim": {
                    key: torch.from_numpy(obs_dict[key]).float().to(self.device)
                    for key in obs_dict
                    if "eef" in key or "gripper" in key
                },
            }
        }

    def load_model(self, ckpt_path):
        dataset_config_dir = os.path.dirname(ckpt_path)
        dataset_config_path = os.path.join(dataset_config_dir, "config.json")
        cfg = parse_network_configs(dataset_config_path)
        policy = BCMLP(cfg)
        policy.load_state_dict(torch.load(ckpt_path)["net_params"], strict=True)
        return policy


class EvalRunner:
    def __init__(
        self, robot, model, init_range, freq, num_rollouts, horizon, result_log_path, is_palm
    ):
        self.current_exp_idx = 0
        self.exp_list = list(range(0, num_rollouts))
        self.last_episode_status = None
        self.success_count = 0
        self.result_log_path = result_log_path
        self.robot = robot
        self.net = model
        self.init_range = init_range
        self.horizon = horizon
        self.rate = rospy.Rate(freq)
        self.palm = is_palm
        self.reset = False

        self.gripper_act_prev = 1
        self.steps = 0
        self.task_id = 0

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
        print("Keyboard listener started (press s/f/r/q)...")

        def on_press(key):
            try:
                if key.char == "s":
                    print("Key 's' pressed: marking as SUCCESS")
                    self.last_episode_status = "success"
                    self.record_last_episode_result()
                    self.success_count += 1
                    self.reset = True

                elif key.char == "f":
                    print("Key 'f' pressed: marking as FAIL")
                    self.last_episode_status = "fail"
                    self.record_last_episode_result()
                    self.reset = True

                elif key.char == "r":
                    print("Key 'r' pressed: marking as RESET")
                    self.last_episode_status = "reset"
                    self.reset = True

                elif key.char == "q":
                    print("Key 'q' pressed: exiting")
                    self.robot.home_robot()
                    rospy.signal_shutdown("User requested shutdown")

                if self.current_exp_idx > 0:
                    success_rate = (self.success_count / self.current_exp_idx) * 100
                    print(
                        f"Success rate: {self.success_count}/{self.current_exp_idx} = {success_rate:.1f}%"
                    )
                    self.robot.home_robot()
                    rospy.signal_shutdown("User requested shutdown")

            except AttributeError:
                pass  # Ignore special keys

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def run_eval(self):
        while self.steps < self.horizon:
            # if keyboard.is_pressed('q'):
            #     print("Exiting...")
            if self.reset:
                self.robot.home_robot()
                self.steps = 0
                self.gripper_act_prev = 1
                self.task_id = 0
                self.reset = False
            if self.steps == 0:
                self.robot.get_obs()
                self.robot.move_robot_to_init(self.task_id, self.init_range)
            if not self.robot.has_states():
                continue
            obs = self.robot.get_obs()
            obs_dict = {
                **obs["rgb_frame"],
                **obs["robot_state"],
                **obs["gripper_state"],
            }
            obs_dict.pop("front_rgb_raw")
            eef_pose = obs_dict.pop("eef_pose")
            if self.palm:
                # obs_dict.pop("eef_6d_xyz")
                obs_dict.pop("eef_6d_in_cam_z")
                obs_dict.pop("front_rgb")
            else:
                obs_dict.pop("front_rgb_overlay_tcp_crop")
                obs_dict.pop("eef_6d_in_cam_z")

            if self.robot.get_gripper_state() < 400 and self.task_id != 1:
                self.task_id = 1
                self.robot.move_robot_to_init(self.task_id, self.init_range)
                print("moving to next task")
            else:
                bot_act, gripper_act = self.net.infer_action(obs_dict)
                action_to_execute = eef_pose @ bot_act
                action_to_execute = TUtils.SE3_to_Eular(action_to_execute).flatten()
                action_to_execute[:3] = action_to_execute[:3] * 1000
                self.robot.move_to_pose(action_to_execute.tolist())
                if self.gripper_act_prev != gripper_act:
                    if gripper_act > 0.5:
                        self.robot.move_gripper(850)
                    else:
                        self.robot.move_gripper(330)
                        rospy.sleep(0.1)
                self.gripper_act_prev = gripper_act
                self.steps += 1
            self.rate.sleep()
            del obs
        rospy.signal_shutdown("User requested shutdown")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PALM Evaluation ROS Node")
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True, help="Horizon for the evaluation")
    parser.add_argument("--freq", type=int, default=5)
    parser.add_argument("--num_rollouts", type=int, default=30)
    parser.add_argument("--init_range", type=float, default=0.06)
    parser.add_argument("-p", "--palm", action="store_true")
    args = parser.parse_args()

    rospy.init_node("palm_eval_ros", anonymous=True)
    xarm_robot = XArmRobot(
        img_size=(200, 200),
        crop_kwargs={
            "tcp_crop_size": (240, 240),
            "tcp_height_offset": 60,
        },
    )
    net = NetWrapper(args.model_path)
    eval_runner = EvalRunner(
        xarm_robot,
        net,
        args.init_range,
        args.freq,
        args.num_rollouts,
        args.horizon,
        result_log_path="./eval_result.txt",
        is_palm=args.palm,
    )

    keyboard_thread = threading.Thread(target=eval_runner.keyboard_listener, daemon=True)
    keyboard_thread.start()

    eval_thread = threading.Thread(target=eval_runner.run_eval, daemon=True)
    eval_thread.start()

    try:
        while not rospy.is_shutdown():
            time.sleep(0.1)  # Keeps main thread alive
    except KeyboardInterrupt:
        print("Shutting down...")
        rospy.signal_shutdown("User requested shutdown")
