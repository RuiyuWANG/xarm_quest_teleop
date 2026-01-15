import numpy as np
import cv2
import PIL.Image
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import (
    MoveJoint,
    MoveCartesian,
    GripperMove,
    SetInt16,
    SetFloat32,
    GetFloat32,
)
from message_filters import ApproximateTimeSynchronizer, Subscriber

import palm.utils.transform_utils as TUtils
import palm.utils.image_utils as PalmUtils
from palm.utils.real_exp_image_utils import (
    project_ee_axes,
    get_object_mask,
    get_object_position_from_mask,
    cam_pt_to_table_pt,
    get_new_cam_matrix_in_a,
)

# Hardcoded parameters
HOME_JOINT = np.deg2rad([0, -45, 0, 45, 0, 90, 0]).tolist()
GRIPPER_HOME_POS = 850.0  # Pulse position for gripper home state
IMAGE_TOPIC = "/camera/camera/color/image_raw"
ROBOT_TOPIC = "/xarm/robot_states"

INTRINSICS_FILE = "charuco_intrinsics.npz"
HAND_EYE_FILE = "handeye_result.npz"
EYE_HAND_FILE = "eye_hand_result.npz"

handeye_data = np.load(HAND_EYE_FILE)
eyehand_data = np.load(EYE_HAND_FILE)
intr_data = np.load(INTRINSICS_FILE)
X_C = handeye_data["base_T_cam"]
T = eyehand_data["EE_T_cam"]
K = intr_data["K"]

class XArmRobot(Node):
    def __init__(self, img_size, crop_kwargs, rot_angle):
        super().__init__("xarm_robot_node")

        self.image_sub = Subscriber(self, Image, IMAGE_TOPIC)
        self.robot_sub = Subscriber(self, RobotMsg, ROBOT_TOPIC)
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.robot_sub], queue_size=10, slop=0.01
        )

        self.set_xarm_mode_client = self.create_client(SetInt16, "/xarm/set_mode")
        self.set_xarm_state_client = self.create_client(SetInt16, "/xarm/set_state")
        self.set_gripper_mode_client = self.create_client(SetInt16, "/xarm/set_gripper_mode")
        self.set_gripper_enable_client = self.create_client(SetInt16, "/xarm/set_gripper_enable")
        self.set_gripper_speed_client = self.create_client(SetFloat32, "/xarm/set_gripper_speed")
        self.move_joint_client = self.create_client(MoveJoint, "/xarm/set_servo_angle")
        self.move_line_client = self.create_client(MoveCartesian, "/xarm/set_position")
        self.gripper_move_client = self.create_client(GripperMove, "/xarm/set_gripper_position")
        self.gripper_state_client = self.create_client(GetFloat32, "/xarm/get_gripper_position")

        for client in [
            self.set_xarm_mode_client,
            self.set_xarm_state_client,
            self.set_gripper_mode_client,
            self.set_gripper_enable_client,
            self.set_gripper_speed_client,
            self.move_joint_client,
            self.move_line_client,
            self.gripper_move_client,
            self.gripper_state_client,
        ]:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for service {client.srv_name}...")

        self.img_size = img_size
        self.crop_kwargs = crop_kwargs
        if rot_angle != 0:
            self.X_C = get_new_cam_matrix_in_a(
                np.deg2rad(rot_angle), X_C
            )
        self.rot_angle = rot_angle

        self.rgb_frame = None
        self.robot_state = None
        self.gripper_state = None
        self.raw_robot_state = None
        self.target_object_pose = None
        self.gripper_raw_pos = None

        self.pose_mode = 0
        self.joint_mode = 0
        self.current_mode = -1

        print("[XArm] Setting up xArm...")
        self.setup_xarm(self.joint_mode, 0)
        print("[XArm] Setting up gripper...")
        self.setup_gripper(enable=True, mode=0, speed=1500.0)
        self.home_robot()
        # start the subscribers
        self.frame_tick = 0
        self.sync.registerCallback(self.callback)
        print("[XArm] Setup complete.\n")

    def callback(self, image_msg, robot_msg):
        # print(f"[Callback] Frame tick: {self.frame_tick}")
        self.frame_tick += 1
        robot_state = self.format_robot_state(robot_msg)
        rgb_frame = self.format_rgb_frame(image_msg, robot_state["eef_pose"])
        gripper_state = self.format_gripper_state()
        raw_robot_state = np.array(robot_msg.pose, dtype=np.float32).reshape(6)
        # if self.target_object_pose is not None:
        # camera_T_EE = np.linalg.inv(self.X_C) @ robot_state["eef_pose"]
        # imgpts = project_ee_axes(camera_T_EE, K)
        # img_to_display = rgb_frame["front_rgb_raw"].copy()
        # cv2.line(img_to_display, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)
        # cv2.line(img_to_display, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3)
        # cv2.line(img_to_display, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3)
        # cv2.imshow("Object Pose Projection", cv2.cvtColor(img_to_display, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)
        self.robot_state = robot_state
        self.rgb_frame = rgb_frame
        self.gripper_state = gripper_state
        self.raw_robot_state = raw_robot_state    
        self.save_debug_images(f"../real_expriment_imgs/{self.frame_tick}.png")    

    def save_debug_images(self, path):
        img_to_display = self.rgb_frame["front_rgb_raw"]
        # img_to_display = np.transpose(img_to_display, (1, 2, 0)) * 255.0
        img_to_display = img_to_display.astype(np.uint8)
        img_to_display = cv2.cvtColor(img_to_display, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_to_display)

    def format_robot_state(self, robot_msg):
        robot_pose = np.array(robot_msg.pose, dtype=np.float32).reshape(6)
        rot = R.from_euler("xyz", robot_pose[3:6]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = robot_pose[:3] / 1000

        eef_6d_xyz = TUtils.SE3_to_6D_xyz(T)
        C_X_H = np.linalg.inv(self.X_C) @ T
        X_Hybrid = C_X_H.copy()
        X_Hybrid[:3, 3] = T[:3, 3]
        eef_6d_in_cam_z = TUtils.SE3_to_6D_z(X_Hybrid)

        return {"eef_pose": T, "eef_6d_xyz": eef_6d_xyz, "eef_6d_in_cam_z": eef_6d_in_cam_z}

    def format_rgb_frame(self, image_msg, ee_pose):
        raw_rgb_frame = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            image_msg.height, image_msg.width, 3
        )
        h, w = raw_rgb_frame.shape[:2]
        margin = (w - h) // 2
        if margin >= 0:
            rgb_frame = raw_rgb_frame[:, margin : margin + h]
        else:
            rgb_frame = raw_rgb_frame
        rgb_resized = (
            np.transpose(
                np.array(PIL.Image.fromarray(rgb_frame).resize(self.img_size)), (2, 0, 1)
            ).astype(np.float32)
            / 255
        )

        cropped = self.get_cropped_image(raw_rgb_frame, self.crop_kwargs, ee_pose)
        resized_cropped = (
            np.transpose(
                np.array(PIL.Image.fromarray(cropped).resize(self.img_size)), (2, 0, 1)
            ).astype(np.float32)
            / 255
        )

        return {
            "front_rgb_raw": raw_rgb_frame,
            "front_rgb": rgb_resized,
            "front_rgb_overlay_tcp_crop": resized_cropped,
        }

    def get_cropped_image(self, image, crop_kwargs, ee_pose):
        image_overlay = PalmUtils.overlay_poses(image, ee_pose, K, self.X_C, axis_length=0.08)
        coords = PalmUtils.project_points(K, self.X_C, poses=ee_pose, to_int=True)
        image_crop = PalmUtils.crop_at_coords(
            image_overlay,
            coords,
            z=ee_pose[2, 3].reshape(1, 1),
            crop_size=crop_kwargs["tcp_crop_size"],
            height_offset=crop_kwargs["tcp_height_offset"],
        )
        return image_crop

    def format_gripper_state(self, open_threshold=400):
        req = GetFloat32.Request()
        future = self.gripper_state_client.call_async(req)
        future.add_done_callback(self.handle_gripper_state_response)
        if self.gripper_raw_pos is None:
            gripper_state = -1.0  # Error state
        else:
            gripper_state = 1.0 if self.gripper_raw_pos > open_threshold else 0.0
        return {"gripper_state": np.array([gripper_state])}

    def handle_gripper_state_response(self, future):
        try:
            response = future.result()
            self.gripper_raw_pos = response.data
        except Exception as e:
            self.get_logger().error(f"[handle_gripper_state_response] Service call failed: {e}")

    def get_obs(self):
        return {
            "rgb_frame": self.rgb_frame,
            "robot_state": self.robot_state,
            "gripper_state": self.gripper_state,
        }

    def move_to_joint(self, joint_pos, wait: bool = True):
        if self.current_mode != self.joint_mode:
            self.setup_xarm(self.joint_mode, 0)
        req = MoveJoint.Request()
        req.angles = joint_pos
        req.speed = 0.35
        req.acc = 10.0
        req.wait = wait
        future = self.move_joint_client.call_async(req)
        self.spin_till_done(future)

    def move_to_pose(
        self, target_pose, wait: bool = True, speed: float = 200.0, acc: float = 1000.0
    ):
        if self.current_mode != self.pose_mode:
            self.setup_xarm(self.pose_mode, 0)
        req = MoveCartesian.Request()
        req.pose = [float(coord) for coord in target_pose]
        req.speed = float(speed)
        req.acc = float(acc)
        req.wait = wait
        future = self.move_line_client.call_async(req)
        self.spin_till_done(future)

    def move_gripper(self, gripper_pos: float, wait: bool = False):
        req = GripperMove.Request()
        req.pos = gripper_pos
        req.wait = wait
        future = self.gripper_move_client.call_async(req)
        self.spin_till_done(future)

    def get_robot_init_state(self, id, rand_range, table_height=0.037, external_image=None):
        img = self.rgb_frame["front_rgb_raw"] if external_image is None else external_image
        object_mask = get_object_mask(img, id=id, task="stack_two")
        object_position_in_pixel = get_object_position_from_mask(object_mask)
        object_xy = cam_pt_to_table_pt(object_position_in_pixel, table_height)
        angles = np.deg2rad([178.485864, 0.001318, 1.393949])
        init_pose = np.array([0.0, 0.0, 160, 0.0, 0.0, 0.0])
        init_pose[:2] = object_xy[:2]
        init_pose[3:] = angles
        if rand_range != 0.00:
            offset = np.random.uniform(-rand_range, rand_range, size=2)
            init_pose[:2] += offset
        init_pose[:2] *= 1000  # Convert to mm
        gripper_pos = 850.0 if id == 0 else 300.0
        return object_xy, init_pose.tolist(), gripper_pos
    
    def get_robot_init_rotation(self):
        if self.rot_angle != 0:
            X_H_prime = self.robot_state["eef_pose"].copy()
            C_X_H_prime = np.linalg.inv(X_C) @ X_H_prime
            X_H_double_prime = self.X_C @ C_X_H_prime
            X_H_prime[:3, :3] = X_H_double_prime[:3, :3]
            
            X_H_prime_to_execute = TUtils.SE3_to_Eular(X_H_prime).flatten()
            X_H_prime_to_execute[:3] = X_H_prime_to_execute[:3] * 1000  # Convert to mm
            return X_H_prime_to_execute
        else:
            return None
    
    def move_to_init_state(self, init_pose, gripper_pos, wait: bool = True):
        self.move_to_pose(init_pose, wait=wait)
        self.move_gripper(gripper_pos, wait=wait)

    def setup_xarm(self, mode: int = 7, state: int = 0):
        req = SetInt16.Request()
        req.data = mode
        future = self.set_xarm_mode_client.call_async(req)
        self.spin_till_done(future)

        req = SetInt16.Request()
        req.data = state
        future = self.set_xarm_state_client.call_async(req)
        self.spin_till_done(future)
        self.current_mode = mode

    def home_robot(self):
        self.move_to_joint(HOME_JOINT)
        print("[XArm] Robot homed to joint position.")
        self.move_gripper(GRIPPER_HOME_POS)
        print("[XArm] Gripper homed to position.")

    def setup_gripper(self, enable: bool = True, mode: int = 0, speed: float = 1500.0):
        req_enable = SetInt16.Request()
        req_enable.data = 1 if enable else 0
        future_enable = self.set_gripper_enable_client.call_async(req_enable)
        self.spin_till_done(future_enable)

        req_mode = SetInt16.Request()
        req_mode.data = mode
        future_mode = self.set_gripper_mode_client.call_async(req_mode)
        self.spin_till_done(future_mode)

        req_speed = SetFloat32.Request()
        req_speed.data = speed
        future_speed = self.set_gripper_speed_client.call_async(req_speed)
        self.spin_till_done(future_speed)

    def spin_till_done(self, future):
        while not future.done() and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        if future.exception():
            self.get_logger().error(f"[spin_till_done] Service call failed: {future.exception()}")

    def has_states(self):
        return (
            self.rgb_frame is not None
            and self.robot_state is not None
            and self.gripper_state is not None
            and self.raw_robot_state is not None
        )


def main():
    rclpy.init()
    img_size = (224, 224)
    crop_kwargs = {"tcp_crop_size": (128, 128), "tcp_height_offset": 0.02}
    robot = XArmRobot(img_size, crop_kwargs)

    for _ in range(10):
        rclpy.spin_once(robot, timeout_sec=0.1)

    try:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
