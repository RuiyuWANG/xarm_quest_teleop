from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import (
    Move,
    MoveRequest,
    GripperMove,
    GripperMoveRequest,
    SetInt16,
    GripperState,
)
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
import rospy
import palm.utils.transform_utils as TUtils
import palm.utils.image_utils as PalmUtils
from img_utils import (
    project_ee_axes,
    get_object_mask,
    get_object_position_from_mask,
    cam_pt_to_table_pt,
)
import cv2
import os
import sys
import numpy as np
import time
from datetime import datetime
import threading
import PIL.Image
import PIL.ImageDraw
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


# Hardcoded parameters
HOME_JOINT = np.deg2rad([0, -45, 0, 45, 0, 90, 0]).tolist()
IMAGE_TOPIC = "/camera/color/image_raw"
ROBOT_TOPIC = "/xarm/xarm_states"

# camera matrices
INTRINSICS_FILE = "charuco_intrinsics.npz"
HAND_EYE_FILE = "handeye_result.npz"

handeye_data = np.load(HAND_EYE_FILE)
intr_data = np.load(INTRINSICS_FILE)

X_C = handeye_data["base_T_cam"]
K = intr_data["K"]


class XArmRobot:
    def __init__(self, img_size, crop_kwargs):
        # Subscribers for image and robot state
        self.image_sub = Subscriber(IMAGE_TOPIC, Image)
        self.robot_sub = Subscriber(ROBOT_TOPIC, RobotMsg)

        self.rgb_frame = None
        self.robot_state = None
        self.gripper_state = None
        self.raw_robot_state = None
        self.gripper_qpos = None
        self.target_object = None
        self.fetch_new_states = False

        # Synchronize the subscribers
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.robot_sub], queue_size=5, slop=0.001
        )
        self.sync.registerCallback(self.callback)

        # policy related parameters
        # HARDCODED
        self.img_size = img_size
        self.crop_kwargs = crop_kwargs

        self.set_mode = rospy.ServiceProxy("/xarm/set_mode", SetInt16)
        self.set_state = rospy.ServiceProxy("/xarm/set_state", SetInt16)
        rospy.set_param("/xarm/wait_for_finish", True)
        self.set_mode(0)
        self.set_state(0)
        self.home_robot()
        self.state_buffer = []

    def clear_states(self):
        self.rgb_frame = None
        self.robot_state = None
        self.gripper_state = None
        self.fetch_new_states = False

    def has_states(self):
        return (
            self.rgb_frame is not None
            # and self.robot_state is not None
            and self.gripper_state is not None
        )

    def callback(self, image_msg, robot_msg):
        if not self.fetch_new_states:
            return
        # Process the image and robot state
        robot_state = self.format_robot_state(robot_msg)
        rgb_frame = self.format_rgb_frame(image_msg, robot_state["eef_pose"])
        gripper_state = self.format_gripper_state()
        raw_robot_state = np.array(robot_msg.pose, dtype=np.float32).reshape(6)

        if self.target_object is not None:
            camera_T_EE = np.linalg.inv(X_C) @ self.target_object
            imgpts = project_ee_axes(camera_T_EE, K)
            img_to_display = self.rgb_frame["front_rgb_raw"].copy()
            cv2.line(img_to_display, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # X - Red
            cv2.line(
                img_to_display, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3
            )  # Y - Green
            cv2.line(
                img_to_display, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3
            )  # Z - Blue

            cv2.imshow("Object Pose Projection", img_to_display)
            cv2.imwrite("test.png", self.rgb_frame["front_rgb_raw"].copy())
            cv2.waitKey(1)
        self.state_buffer.append(robot_state)
        self.rgb_frame = rgb_frame
        self.gripper_state = gripper_state
        self.raw_robot_state = raw_robot_state

        # current_image = self.rgb_frame["front_rgb_raw"]
        # bgr_frame = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
        # def on_mouse_click(event, x, y, flags, param):
        #     if event == cv2.EVENT_LBUTTONDOWN and current_image is not None:
        #         bgr = bgr_frame[y, x]
        #         hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        #         print(f"[{x}, {y}] BGR: {bgr} → HSV: {hsv}")
        # object_mask = get_object_mask(
        #     current_image, id=1, task="stack_two")robot_state,


        # cv2.namedWindow("Click to Get HSV")
        # cv2.setMouseCallback("Click to Get HSV", on_mouse_click)
        # cv2.imshow("object_mask",(object_mask).astype(np.uint8))
        # cv2.waitKey(1)

    def format_gripper_state(self):
        # TODO: Double check this shit
        self.gripper_qpos = self.get_gripper_state()
        gripper_state = 1.0 if self.gripper_qpos > 380 else 0.0
        return {"gripper_state": np.array([gripper_state])}

    def format_robot_state(self, robot_msg):
        robot_pose = np.array(robot_msg.pose, dtype=np.float32).reshape(6)
        rot = R.from_euler("xyz", robot_pose[3:6]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = robot_pose[:3] / 1000

        eef_6d_xyz = TUtils.SE3_to_6D_xyz(T)
        C_X_H = np.linalg.inv(X_C) @ T
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
        rgb_resized = self.resize_image(rgb_frame, self.img_size)
        rgb_resized = np.transpose(rgb_resized, (2, 0, 1)).astype(np.float32) / 255

        cropped = self.get_cropped_image(raw_rgb_frame, self.crop_kwargs, ee_pose)
        resized_cropped = self.resize_image(cropped, self.img_size)
        resized_cropped = np.transpose(resized_cropped, (2, 0, 1)).astype(np.float32) / 255

        # time = datetime.now().strftime("%Y%m%d%H%M%S%f")
        # PIL.Image.fromarray(cropped).save(f"../data/debug_imgs/{time}.png")
        return {
            "front_rgb_raw": raw_rgb_frame,
            "front_rgb": rgb_resized,
            "front_rgb_overlay_tcp_crop": resized_cropped,
        }

    def resize_image(self, image: np.ndarray, size: tuple) -> np.ndarray:
        pil_img = PIL.Image.fromarray(image)
        return np.array(pil_img.resize(size))

    def get_cropped_image(
        self, image: np.ndarray, crop_kwargs: dict, ee_poses: np.ndarray
    ) -> np.ndarray:
        # TODO: no tcp overlay at the moment
        # HACK
        # ee_poses_adjusted = ee_poses.copy()
        # ee_poses_adjusted[2, 3] -= 0.003
        image_overlay = PalmUtils.overlay_poses(
            images=image, poses=ee_poses, K=K, X_C=X_C, axis_length=0.08
        )
        coords = PalmUtils.project_points(K=K, X_C=X_C, poses=ee_poses, to_int=True)
        image_crop = PalmUtils.crop_at_coords(
            images=image_overlay,
            coords=coords,
            z=ee_poses[2, 3].reshape(1, 1),
            crop_size=crop_kwargs["tcp_crop_size"],
            height_offset=crop_kwargs["tcp_height_offset"],
        )

        return image_crop

    def home_robot(self):
        self.move_to_joint(HOME_JOINT)
        self.move_gripper(850)

    def move_to_pose(self, target_pose):
        assert isinstance(target_pose, list) and len(target_pose) == 6
        move_line = rospy.ServiceProxy("/xarm/move_line", Move)
        req = MoveRequest()
        req.pose = target_pose
        # TODO: double check velocity and acceleration
        req.mvvelo = 0
        req.mvacc = 0
        req.mvtime = 0
        try:
            res = move_line(req)
            if res.ret:
                print("Something Wrong happened calling move_line service, ret = %d" % res.ret)
                ret = -1
                return ret
        except rospy.ServiceException as e:
            print("move_line Service call failed: %s" % e)
            return -1

    def move_to_joint(self, joint_pos):
        assert isinstance(joint_pos, list) and len(joint_pos) == 7
        # move to joint position
        move_joint = rospy.ServiceProxy("/xarm/move_joint", Move)
        req = MoveRequest()
        req.pose = joint_pos
        req.mvvelo = 0
        req.mvacc = 0
        req.mvtime = 0
        try:
            res = move_joint(req)
            if res.ret:
                print("Something Wrong happened calling move_joint service, ret = %d" % res.ret)
                ret = -1
                return ret
        except rospy.ServiceException as e:
            print("move_joint Service call failed: %s" % e)
            return -1

    def get_gripper_state(self):
        try:
            ret = gripper_srv()
            return ret.curr_pos
        except rospy.ServiceException as e:
            print("gripper_state Service call failed: %s" % e)
            return -1

    def move_gripper(self, gripper_pos):
        assert isinstance(gripper_pos, (int, float)) and -100 <= gripper_pos <= 850
        rospy.wait_for_service("/xarm/gripper_move")
        try:
            gripper_serv = rospy.ServiceProxy("/xarm/gripper_move", GripperMove)
            req = GripperMoveRequest()
            req.pulse_pos = float(gripper_pos)  # must be float32
            res = gripper_serv(req)

            if res.ret != 0:
                print(f"[gripper_move] Failed with ret={res.ret}, message: {res.message}")
                return -1
            print(f"[gripper_move] Success: {res.message}")
            return 0
        except rospy.ServiceException as e:
            print(f"[gripper_move] Service call failed: {e}")
            return -1

    def move_robot_to_init(self, id, rand_range, table_height=0.037):
        object_mask = get_object_mask(self.rgb_frame["front_rgb_raw"], id=id, task="stack_two")

        object_position_in_pixel = get_object_position_from_mask(object_mask)
        object_position = cam_pt_to_table_pt(object_position_in_pixel, table_height)

        x, y = object_position[0], object_position[1]
        if rand_range != 0.00:
            offset = np.random.uniform(-rand_range, rand_range, size=2)
            x = x + offset[0]
            y = y + offset[1]
        angles = np.deg2rad([178.485864, 0.001318, 1.393949])
        init_pose = [x * 1000, y * 1000, 160, angles[0], angles[1], angles[2]]
        self.move_to_pose(init_pose)

        gripper_pos = 850 if id == 0 else 300
        self.move_gripper(gripper_pos)
        return object_position

    def get_obs(self):
        self.clear_states()
        self.fetch_new_states = True
        max_attempts = 100
        attemps = 0
        while not self.has_states() and attemps < max_attempts:
            rospy.sleep(0.01)  # Sleep briefly to avoid busy-waiting
            attemps += 1
        if attemps == max_attempts:
            raise RuntimeError("Failed to fetch states after {} attempts".format(max_attempts))
        return {
            "rgb_frame": self.rgb_frame,
            "robot_state": self.state_buffer[-5] if len(self.state_buffer) > 5 else self.state_buffer[-1],
            "gripper_state": self.gripper_state,
        }

    def run_motion(self):
        """
        Moves the robot in a demo loop while the viz thread is running.
        You can customize this with more complex motion.
        """
        joint_positions = [
            HOME_JOINT,
            np.deg2rad([5, -47, 10, 30, 0, 90, 0]).tolist(),
            np.deg2rad([-5, -43, -10, 30, 0, 90, 0]).tolist(),
            HOME_JOINT,
        ]
        i = 0
        while not rospy.is_shutdown():
            print(f"[motion] Moving to pose {i % len(joint_positions)}")
            self.move_to_joint(joint_positions[i % len(joint_positions)])
            time.sleep(2)  # wait before next move
            i += 1


# def test():
#     rospy.init_node("palm_eval_ros", anonymous=True)
#     xarm_robot = XArmRobot()

#     # Start robot motion in a separate thread
#     motion_thread = threading.Thread(target=xarm_robot.run_motion)
#     motion_thread.start()

# if __name__ == "__main__":
#     test()
