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
WORKSPACE_STACK = [0.228, -0.13, 0.12, 0.12]

# Camera matrices
INTRINSICS_FILE = "charuco_intrinsics.npz"
HAND_EYE_FILE = "handeye_result.npz"
handeye_data = np.load(HAND_EYE_FILE)
X_C = handeye_data["base_T_cam"]
intr_data = np.load(INTRINSICS_FILE)
K = intr_data["K"]

class DemoCollector:
    def __init__(self, save_dir, num_demos=100, init_range=0.1, freq=10):
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
        self.robot = XArmAPI(ROBOT_IP)
        self.robot.motion_enable(enable=True)
        self.robot.set_gripper_enable(enable=True)
        self.robot.set_mode(0)
        self.robot.set_state(state=0)
        self.robot.set_gripper_mode(mode=0)

        # Move to initial position
        self.home()

    def keyboard_listener(self):
        print("Keyboard listener started (press b/s/c/q)...")

        def on_press(key):
            try:
                if key.char == "b":
                    target_object  = self.move_robot_to_init()
                    
                    target_obj_pose = np.eye(4)
                    R_curr = self.get_eef_pose()[:3, :3]
                    obj_rotation = sample_ee_xy_rotation(R_curr, angle_range_deg=(-45, 45))
                    new_z  = (0.035 + np.random.uniform(-0.015, 0.015))
                    # new_xy = sample_pose_within_workspace(x0=WORKSPACE_STACK[0], y0=WORKSPACE_STACK[1], x_bounds=WORKSPACE_STACK[2], y_bounds=WORKSPACE_STACK[3], slack=0.02)
                    target_obj_pose[:3, :3] = obj_rotation
                    target_obj_pose[:2, 3] = target_object[:2] / 1000
                    target_obj_pose[2, 3] = new_z
                    self.target_obj_pose = target_obj_pose
                    
                    print(f"[INFO] Begin demo collection {self.current_demo_idx + 1}/{self.num_demos}, subtask {self.id}")

                elif key.char == "s":
                    self.mixed_trajectory_replay_and_record()
                    print(f"[INFO] Demo {self.current_demo_idx + 1} saved")
                    
                    self.current_demo_idx += 1
                    if self.current_demo_idx >= self.num_demos:
                        print("[INFO] All demos collected.")
                        return
                    
                    self.id = 0
                    self.last_z = 0
                    self.gripper_open = True
                    self.image = []
                    self.target_obj_pose = np.eye(4)
                    self.home()

                elif key.char == "c":
                    print("[INFO] Demo canceled.")
                    
                    self.id = 0
                    self.last_z = 0
                    self.gripper_open = True
                    self.image = []
                    self.target_obj_pose = np.eye(4)
                    self.home()

                elif key.char == "q":
                    print("[INFO] Quit requested.")
                    self.stop_recording.set()
                    self.home()
                    return False

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
                
                # go x y
                inter_z = 110 + np.random.uniform(-0.015, 0.015) * 1000
                self.robot.set_position(x=self.target_obj_pose[0, 3]*1000, y=self.target_obj_pose[1, 3]*1000, z=inter_z, wait=True, is_radian=False)
                
                # rotation, go down, close gripper
                new_rot = TUtils.SE3_to_Eular(self.target_obj_pose)
                new_z = self.target_obj_pose[2, 3] * 1000
                self.last_z = new_z
                self.robot.set_position(roll=new_rot[3], pitch=new_rot[4], yaw=new_rot[5], z=new_z, wait=True, is_radian=True)
                self.robot.set_gripper_position(330, wait=True)
                
                # update status
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
        
        # Record and execute subtask0 (meta policy for grasping)
        self.stop_recording.clear()
        recording_thread0 = threading.Thread(target=self.record_demo, args=(traj0_path,))
        recording_thread0.start()
        
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
                # Get low dim data
                ee_pose = self.get_eef_pose()
                raw_eef_pose = self.get_raw_eef()
                gripper_qpos = self.get_gripper_pose()
                
                # process img
                with self.image_lock:
                    if len(self.image) == 0:
                        continue
                    img = self.image[-1].copy()
                h, w = img.shape[:2]
                margin = int(w - h) // 2
                if margin >= 0:
                    img = img[:, margin : margin + h]
                    
                # Save low_dim data
                low_dim_data[count] = {
                    "ee_pose": ee_pose.tolist(),
                    "gripper_qpos": float(gripper_qpos),
                    "raw_eef_pose": raw_eef_pose.tolist()
                }

                # Save image
                img_path = os.path.join(save_path, "images", f"{count}.png")
                cv2.imwrite(img_path, img)
                
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
                    
                # Project target object pose
                ee_pose = self.target_obj_pose
                camera_T_EE = np.linalg.inv(X_C) @ ee_pose
                imgpts = project_ee_axes(camera_T_EE, K)
                img_to_display = agentview_image.copy()
                cv2.line(img_to_display, tuple(imgpts[0]), tuple(imgpts[1]), (0,0,255), 3)  # X - Red
                cv2.line(img_to_display, tuple(imgpts[0]), tuple(imgpts[2]), (0,255,0), 3)  # Y - Green
                cv2.line(img_to_display, tuple(imgpts[0]), tuple(imgpts[3]), (255,0,0), 3)  # Z - Blue
                
                cv2.imshow("Object Pose Projection", img_to_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(f"[image_stream_loop] Error: {e}")
                time.sleep(0.1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="../data/real_stack_auto")
    parser.add_argument("--num_demos", type=int, default=150)
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--init_range", type=float, default=0.08)
    args = parser.parse_args()

    demo_collector = DemoCollector(
        save_dir=args.save_dir,
        num_demos=args.num_demos,
        init_range=args.init_range,
        freq=args.freq,
    )
    
    # Start image stream thread
    image_thread = threading.Thread(target=demo_collector.image_stream_loop, daemon=True)
    image_thread.start()
    
    # Wait for first image
    time.sleep(1.0)
    
    # Start keyboard listener
    demo_collector.keyboard_listener()