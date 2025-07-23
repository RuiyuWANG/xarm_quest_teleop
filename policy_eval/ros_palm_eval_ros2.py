import os
import time

import rclpy
import torch
from asset.robot_ros2 import XArmRobot

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
        self.palm = is_palm
        self.reset = False

        self.gripper_act_prev = 1
        self.steps = 0
        self.task_id = 0
        self.init_rpy = None
        self.quit_requested = False
        self.continue_trajectory = False
        self.frame_for_mask = None

        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    def record_last_episode_result(self):
        if self.current_exp_idx >= len(self.exp_list):
            return
        exp_id = self.exp_list[self.current_exp_idx]
        line = f"{exp_id}: {self.last_episode_status}\n"
        os.makedirs(os.path.dirname(self.result_log_path), exist_ok=True)

        with open(self.result_log_path, "a") as f:
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
                elif key.char == "c":
                    print("Key 'c' pressed: continue")
                    self.continue_trajectory = True

                elif key.char == "q":
                    print("Key 'q' pressed: quitting")
                    self.quit_requested = True

                print(f"Success Rate: {self.success_count} / {self.current_exp_idx}")

            except AttributeError:
                pass  # Ignore special keys

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def run_eval_step(self):
        if not self.continue_trajectory:
            return
        
        if self.steps >= self.horizon:
            print("Horizon reached, shutting down.")
            self.reset = True
            return

        if not self.robot.has_states():
            return

        if self.steps == 0:
            self.frame_for_mask = self.robot.rgb_frame["front_rgb_raw"]
            obj_xy, init_xyz_rpy, gripper_pos = self.robot.get_robot_init_state(
                self.task_id, self.init_range, external_image=self.frame_for_mask
            )
            self.init_rpy = init_xyz_rpy[3:6]
            self.robot.move_to_init_state(init_xyz_rpy, gripper_pos)
            # rotate wrt to the camera
            if self.palm and self.robot.rot_angle != 0:
                init_rot_xyz_rpy = self.robot.get_robot_init_rotation()
                self.robot.move_to_init_state(init_rot_xyz_rpy, gripper_pos)

        obs = self.robot.get_obs()
        obs_dict = {
            **obs["rgb_frame"],
            **obs["robot_state"],
            **obs["gripper_state"],
        }
        obs_dict.pop("front_rgb_raw")
        eef_pose = obs_dict.pop("eef_pose")
        if self.palm:
            obs_dict.pop("eef_6d_xyz")
            obs_dict.pop("front_rgb")
        else:
            obs_dict.pop("front_rgb_overlay_tcp_crop")
            obs_dict.pop("eef_6d_in_cam_z")

        if self.robot.gripper_raw_pos < 400 and self.task_id != 1:
            self.task_id = 1
            _, init_xyz_rpy, gripper_pos = self.robot.get_robot_init_state(
                self.task_id, self.init_range, external_image=self.frame_for_mask
            )
            init_xyz_rpy[3:6] = self.init_rpy
            self.robot.move_to_init_state(init_xyz_rpy, gripper_pos)
            
            # rotate wrt to the camera
            if self.palm and self.robot.rot_angle != 0:
                init_rot_xyz_rpy = self.robot.get_robot_init_rotation()
                self.robot.move_to_init_state(init_rot_xyz_rpy, gripper_pos)
        else:
            bot_act, gripper_act = self.net.infer_action(obs_dict)
            action_to_execute = eef_pose @ bot_act
            action_to_execute = TUtils.SE3_to_Eular(action_to_execute).flatten()
            action_to_execute[:3] = action_to_execute[:3] * 1000
            self.robot.move_to_pose(action_to_execute.tolist())
            if self.gripper_act_prev != gripper_act:
                if gripper_act > 0.5:
                    self.robot.move_gripper(850.0)
                else:
                    self.robot.move_gripper(330.0)
                    time.sleep(0.1)

            self.gripper_act_prev = gripper_act
            self.steps += 1


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PALM Evaluation ROS 2 Node")
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--freq", type=int, default=5)
    parser.add_argument("--num_rollouts", type=int, required=True)
    parser.add_argument("--init_range", type=float, default=0.06)
    parser.add_argument("-p", "--palm", action="store_true")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save evaluation results")
    parser.add_argument("--rot_angle", type=int, default=0,
                        help="Rotation angle for the camera with respect to the default position in degrees")
    args = parser.parse_args()

    rclpy.init()
    xarm_robot = XArmRobot(
        img_size=(200, 200),
        crop_kwargs={
            "tcp_crop_size": (240, 240),
            "tcp_height_offset": 60,
        },
        rot_angle=args.rot_angle,
    )

    net = NetWrapper(args.model_path)
    eval_runner = EvalRunner(
        xarm_robot,
        net,
        args.init_range,
        args.freq,
        args.num_rollouts,
        args.horizon,
        result_log_path=args.save_path,
        is_palm=args.palm,
    )

    try:
        while rclpy.ok() and eval_runner.current_exp_idx < args.num_rollouts:
            rclpy.spin_once(xarm_robot, timeout_sec=0.1)
            eval_runner.run_eval_step()
            if eval_runner.reset:
                eval_runner.robot.home_robot()
                eval_runner.steps = 0
                eval_runner.gripper_act_prev = 1
                eval_runner.task_id = 0
                eval_runner.reset = False
                eval_runner.continue_trajectory = False
            if eval_runner.quit_requested:
                xarm_robot.home_robot()
                rclpy.shutdown()

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, shutting down.")
        xarm_robot.home_robot()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
