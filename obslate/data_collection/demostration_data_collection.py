import os
import time
import numpy as np
from pynput import keyboard
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
from img_utils import *
import threading
import json
import palm.utils.transform_utils as TUtils

# Robot ip and home pose (joints)
ROBOT_IP = "192.168.1.244"
HOME = [0, -45, 0, 45, 0, 90, 0]

# Camera matrices
INTRINSICS_FILE = "charuco_intrinsics.npz"
HAND_EYE_FILE = "handeye_result.npz"
handeye_data = np.load(HAND_EYE_FILE)
X_C = handeye_data["base_T_cam"]
intr_data = np.load(INTRINSICS_FILE)
K = intr_data["K"]

class DemoCollector:
    def __init__(self, save_dir, num_demos=100, init_range=0.1, freq=10, is_radian=False):
        self.robot = XArmAPI(ROBOT_IP)
        self.save_dir = save_dir
        self.init_range = init_range
        self.num_demos = num_demos
        self.freq = freq
        self.is_radian = is_radian
        
        self.gripper_open = True
        self.current_demo_idx = 0
        self.id = 0
        self.image = []
        self.image_lock = threading.Lock()
        self.stop_recording = threading.Event()
        self.last_z = 0
        self.start_joint = None
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        subtask0_dir = os.path.join(self.save_dir, "subtask0")
        subtask1_dir = os.path.join(self.save_dir, "subtask1")
        
        if os.path.exists(subtask0_dir) and os.path.exists(subtask1_dir):
            episodes_0 = [
                d for d in os.listdir(subtask0_dir)
                if os.path.isdir(os.path.join(subtask0_dir, d))
            ]
            episodes_1 = [
                d for d in os.listdir(subtask1_dir)
                if os.path.isdir(os.path.join(subtask1_dir, d))
            ]

            assert len(episodes_0) == len(episodes_1), (
                f"Mismatch: subtask0 has {len(episodes_0)} folders, "
                f"subtask1 has {len(episodes_1)} folders."
            )
            self.current_demo_idx = len(episodes_0)
        else:
            self.current_demo_idx = 0
                
        # Initialize camera
        self.camera = setup_camera()

        # Initialize robot
        self.robot.motion_enable(enable=True)
        self.robot.set_gripper_enable(enable=True)
        self.robot.set_mode(0)
        self.robot.set_gripper_mode(mode=0)
        self.robot.set_state(state=0)

        # Move to initial position
        self.home()
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "tmp"), exist_ok=True)

    def keyboard_listener(self):
        print("Keyboard listener started (press b/s/c/q)...")

        def on_press(key):
            try:
                if key.char == "b":
                    _  = self.move_robot_to_init()
                    _, start_joint = self.robot.get_servo_angle(servo_id=8, is_radian=False)
                    self.start_joint = start_joint
                    time.sleep(1)
                    self.robot.set_mode(2)
                    self.robot.set_state(0)
                    self.robot.start_record_trajectory()
                    print(f"[INFO] Begin demo collection {self.current_demo_idx + 1}/{self.num_demos}, subtask {self.id}")

                elif key.char == "s":
                    joint_traj_path = f"demo_{self.current_demo_idx}.traj"
                    self.robot.stop_record_trajectory()
                    self.robot.save_record_trajectory(filename=joint_traj_path, wait=True)
                    self.robot.set_mode(0)
                    self.robot.set_state(0)
                    
                    time.sleep(1)
                    self.mixed_trajectory_replay_and_record()
                    print(f"[INFO] Demo {self.current_demo_idx + 1} saved")
                    self.current_demo_idx += 1
                    
                    if self.current_demo_idx >= self.num_demos:
                        print("[INFO] All demos collected.")
                        return
                    
                    self.id = 0
                    self.last_z = 0
                    self.start_joint = None
                    self.gripper_open = True
                    self.image = []
                    self.home()

                elif key.char == "c":
                    self.robot.stop_record_trajectory()
                    self.robot.set_mode(0)
                    self.robot.set_state(0)
                    print("[INFO] Demo canceled.")
                    
                    self.id = 0
                    self.last_z = 0
                    self.start_joint = None
                    self.gripper_open = True
                    self.image = []
                    self.home()

                elif key.char == "q":
                    print("[INFO] Quit requested.")
                    self.stop_recording.set()
                    self.home()
                    return False  # Stop listener

            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
            
    def get_eef_pose(self):
        code, pose = self.robot.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError("Failed to get robot pose.")
        pose = np.array(pose)
        pose[:3] = pose[:3] / 1000  # Convert mm to m
        X = TUtils.Eular_to_SE3(pose)  # Convert to SE(3)
        return X
    
    def get_raw_eef(self):
        code, pose = self.robot.get_position(is_radian=False)
        if code != 0:
            raise RuntimeError("Failed to get robot pose.")
        pose = np.array(pose)
        pose[:3] = pose[:3] / 1000  # Convert mm to m
        return pose

    def get_gripper_pose(self):
        """Get current gripper qpos as a (1,) array"""
        code, gripper_qpos = self.robot.get_gripper_position()
        if code != 0:
            raise RuntimeError("Failed to get gripper qpose")
        return np.array(gripper_qpos)

    def home(self):
        self.robot.set_servo_angle(servo_id=8, angle=HOME, wait=True, is_radian=False)
        self.robot.set_gripper_position(850, speed=5000, wait=True)

    def move_robot_to_init(self):
        # Wait until an image is available
        while True:
            with self.image_lock:
                if len(self.image) > 0:
                    img = self.image[-1]
                    break
            time.sleep(0.1)
        
        object_mask = get_object_mask(img, id=self.id, task="stack_two")
        object_position_in_pixel = get_object_position_from_mask(object_mask)
        table_height = 0.035
        object_position = cam_pt_to_table_pt(object_position_in_pixel, table_height)

        init_pose = np.array([object_position[0], object_position[1], 160, 178.485864, 0.001318, 1.393949])
        offset = np.random.uniform(-self.init_range, self.init_range, size=2)
        x = init_pose[0] + offset[0]
        y = init_pose[1] + offset[1]
        init_pose = [x*1000, y*1000, init_pose[2], init_pose[3], init_pose[4], init_pose[5]]

        code, joint_angles = self.robot.get_inverse_kinematics(init_pose, input_is_radian=False, return_is_radian=False)
        if code != 0:
            raise RuntimeError(f"Failed to get inverse kinematics: {code}")
        self.robot.set_servo_angle(servo_id=8, angle=joint_angles, wait=True, is_radian=False)

        gripper_pos = 850 if self.gripper_open else 300
        self.robot.set_gripper_position(gripper_pos, wait=True)
        return object_position * 1000
        
    def meta_policy(self, target_position=None, task="stack_two"):
        if task == "stack_two":
            if self.id == 0:  # go down and grasp
                code, curr_eef_pose = self.robot.get_position(is_radian=False)
                if code != 0:
                    raise RuntimeError("Failed to get end-effector pose")
                new_z  = (0.035 + np.random.uniform(-0.015, 0.015)) * 1000
                self.last_z = new_z
                self.robot.set_position(z=new_z, wait=True, is_radian=False)
                self.robot.set_gripper_position(330, wait=True)
                self.id = 1
                self.gripper_open = False
            elif self.id == 1:  # go down and place the cube
                code, curr_eef_pose = self.robot.get_position(is_radian=False)
                if code != 0:
                    raise RuntimeError("Failed to get end-effector pose")
                new_z  = (self.last_z / 1000+ 0.035 + np.random.uniform(0, 0.01)) * 1000
                if target_position is not None:
                    new_eef_pose = [target_position[0], target_position[1]]
                else:
                    new_eef_pose = [curr_eef_pose[0], curr_eef_pose[1]]
                print("target ee", new_eef_pose)
                self.robot.set_position(x=new_eef_pose[0], y=new_eef_pose[1], z=new_z, wait=True, is_radian=False)
                self.robot.set_gripper_position(850, wait=True)
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
        
        if self.start_joint is not None:
            self.robot.set_servo_angle(servo_id=8, angle=self.start_joint, wait=True, is_radian=False)
        time.sleep(0.5)
        
        # Record and execute subtask0 (human demo + meta policy for grasping)
        self.stop_recording.clear()
        recording_thread0 = threading.Thread(target=self.record_demo, args=(traj0_path,))
        recording_thread0.start()
        
        # Replay human trajectory for approach
        joint_traj_path = f"demo_{self.current_demo_idx}.traj"
        self.robot.playback_trajectory(times=1, filename=joint_traj_path, wait=True, double_speed=1)
        
        # Execute meta policy for grasping
        self.meta_policy(task="stack_two")
        
        # Lift after grasping
        code, curr_eef_pose = self.robot.get_position(is_radian=False)
        lift_z = curr_eef_pose[2] + 50  # Lift by 50 mm
        self.robot.set_position(z=lift_z, wait=True, is_radian=False)
        
        # Stop recording for subtask0
        self.stop_recording.set()
        recording_thread0.join()
        
        # Lift after grasping
        self.robot.set_position(z=160, wait=True, is_radian=False)
        
        # Move to placing position (transition - not recorded)
        target_object = self.move_robot_to_init()
        time.sleep(0.1)
        
        # Record and execute subtask1 (meta policy for placing)
        self.stop_recording.clear()
        recording_thread1 = threading.Thread(target=self.record_demo, args=(traj1_path,))
        recording_thread1.start()
        
        # Execute meta policy for placing
        self.meta_policy(target_position=target_object, task="stack_two")
        
        # Lift after placing (transition - not recorded)
        code, curr_eef_pose = self.robot.get_position(is_radian=False)
        lift_z = curr_eef_pose[2] + 50
        self.robot.set_position(z=lift_z, wait=True, is_radian=False)
        
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
                # Get data
                ee_pose = self.get_eef_pose()
                gripper_qpos = self.get_gripper_pose()
                with self.image_lock:
                    if len(self.image) == 0:
                        continue
                    img = self.image[-1].copy()
                raw_eef_pose = self.get_raw_eef()
                
                # process img
                h, w = img.shape[:2]
                margin = int(w - h) // 2
                if margin >= 0:
                    img = img[:, margin : margin + h]

                # Save image
                img_path = os.path.join(save_path, "images", f"{count}.png")
                cv2.imwrite(img_path, img)
                
                # Save low_dim data
                low_dim_data[count] = {
                    "ee_pose": ee_pose.tolist(),
                    "gripper_qpos": float(gripper_qpos),
                    "raw_eef_pose": raw_eef_pose.tolist()
                }
                
                count += 1
            except Exception as e:
                print(f"Recording error: {e}")
        
        # Save low_dim_data to JSON
        with open(os.path.join(save_path, "low_dim.json"), 'w') as f:
            json.dump(low_dim_data, f)
            
    def image_stream_loop(self):
        while True:
            try:
                frames = self.camera.wait_for_frames()
                color_frame = frames.get_color_frame()
                agentview_image = np.asanyarray(color_frame.get_data())
                with self.image_lock:
                    self.image = [agentview_image]
            except Exception as e:
                print(f"[image_stream_loop] Error: {e}")
                time.sleep(0.1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="../data/real_stack")
    parser.add_argument("--num_demos", type=int, default=200)
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--init_range", type=float, default=0.1)
    parser.add_argument("--is_radian", action="store_true")
    args = parser.parse_args()

    demo_collector = DemoCollector(
        save_dir=args.save_dir,
        num_demos=args.num_demos,
        init_range=args.init_range,
        freq=args.freq,
        is_radian=args.is_radian
    )
    
    # Start image stream thread
    image_thread = threading.Thread(target=demo_collector.image_stream_loop, daemon=True)
    image_thread.start()
    
    # Wait for first image
    time.sleep(1.0)
    
    # Start keyboard listener
    demo_collector.keyboard_listener()