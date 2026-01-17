# usage: python camera_calibration.py [~camera_name:=d405] [~setup:=eye_to_hand]
"""
ROS1 Charuco calibration runner (auto-launch robot + ONE selected Realsense).

User inputs (ROS params):
  - ~camera_name : one of {"d405", "d435i_front", "d435i_shoulder"}
  - ~setup       : "eye_to_hand" or "eye_in_hand"

Hardcoded here:
  - realsense launch cmds (by serial + camera namespace)
  - camera image/info topics (by camera_name)
  - output JSON path
  - robot init uses your ROS-based XArmRobot (no xarm.wrapper import)
  - manual mode = 2

Behavior:
  - Auto-launch robot bringup and selected camera driver via ManagedProcess
  - Wait for required topics/services
  - Set robot to manual mode (2)
  - Capture samples with keyboard:
        s : capture (robot pose + charuco pose)
        c : compute calibration + save to JSON
        e : toggle eval overlay
        r : reset samples
        q : quit

Calibration:
  - eye_to_hand (camera fixed, board on EE): solve base_T_cam
  - eye_in_hand (camera on wrist, board fixed on table): solve EE_T_cam
    + for evaluation, estimate base_T_board from captured samples and then
      predict cam_T_board; draw predicted board origin as a red point + pred-vs-obs error.

Notes:
  - Robot pose is taken from XArmRobot.get_state().ee_pose, assumed [x,y,z,roll,pitch,yaw]
    with xyz in mm and rpy in radians (matching your collector code).
"""

import os
import sys
import json
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from pynput import keyboard
from scipy.linalg import expm

# Make your repo importable (same pattern as your script)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.io.process_manager import ManagedProcess, wait_for_topic, wait_for_service
from src.configs.teleop_config import TeleopConfig
from src.configs.robot_config import ROBOT_TOPIC, XArmServices
from src.robots.xarm import XArmRobot


# ===================== Hardcoded outputs / topics / launch cmds =====================
OUT_JSON = "./all_cams_calib.json"
MANUAL_MODE = 2

# Realsense launch commands (exactly from your config snippet)
REALSENSE_CMDS: Dict[str, List[str]] = {
    "d405": [
        "roslaunch", "realsense2_camera", "rs_camera.launch",
        "serial_no:=230322271104", "camera:=d405",
        "enable_color:=true", "enable_depth:=true",
        "filters:=",
    ],
    "d435i_front": [
        "roslaunch", "realsense2_camera", "rs_camera.launch",
        "serial_no:=335522071488", "camera:=d435i_front",
        "enable_color:=true", "enable_depth:=true",
        "enable_gyro:=false", "enable_accel:=false",
        "filters:=",
    ],
    "d435i_shoulder": [
        "roslaunch", "realsense2_camera", "rs_camera.launch",
        "serial_no:=233522073481", "camera:=d435i_shoulder",
        "enable_color:=true", "enable_depth:=true",
        "enable_gyro:=false", "enable_accel:=false",
        "filters:=",
    ],
}

# Camera topics (RGB + CameraInfo). Depth not needed for Charuco pose.
CAMERA_TOPICS: Dict[str, Dict[str, str]] = {
    "d405": {
        "image": "/d405/color/image_raw",
        "info": "/d405/color/camera_info",
    },
    "d435i_front": {
        "image": "/d435i_front/color/image_raw",
        "info": "/d435i_front/color/camera_info",
    },
    "d435i_shoulder": {
        "image": "/d435i_shoulder/color/image_raw",
        "info": "/d435i_shoulder/color/camera_info",
    },
}


# ===================== Charuco config =====================
def aruco_get_dict(dict_id: int):
    # OpenCV 4.2: Dictionary_get
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)

def aruco_detector_params():
    # OpenCV 4.2: DetectorParameters_create
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()

BOARD_SIZE = (7, 5)
SQUARE_LENGTH = 0.035
MARKER_LENGTH = 0.026
ARUCO_DICT = aruco_get_dict(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco_detector_params()


# ===================== SE(3) helpers =====================
def euler_to_se3(pose6, is_radian: bool = True, pos_in_mm: bool = True) -> np.ndarray:
    """
    Convert pose [x, y, z, roll, pitch, yaw] to SE(3).

    Rotation convention:
      R = Rz(yaw) @ Ry(pitch) @ Rx(roll)   (RPY)

    Args:
        pose6: length-6 iterable
        is_radian: angles are radians if True else degrees
        pos_in_mm: xyz are mm if True else meters

    Returns:
        (4,4) SE(3)
    """
    p = np.asarray(pose6, dtype=np.float64).reshape(6,)
    x, y, z, roll, pitch, yaw = p

    if pos_in_mm:
        x, y, z = x / 1000.0, y / 1000.0, z / 1000.0

    if not is_radian:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,    0, 1]], dtype=np.float64)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)

    R = Rz @ Ry @ Rx
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3] = -Rm.T @ t
    return Ti


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3,)
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float64)


def log_SO3(Rm: np.ndarray) -> np.ndarray:
    tr = float(np.trace(Rm))
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if np.isclose(theta, 0.0):
        return np.zeros(3, dtype=np.float64)
    lnR = (theta / (2.0 * np.sin(theta))) * (Rm - Rm.T)
    return np.array([lnR[2, 1], lnR[0, 2], lnR[1, 0]], dtype=np.float64)


def solve_AX_XB(A_list: List[np.ndarray], B_list: List[np.ndarray]) -> np.ndarray:
    """
    Solve A X = X B using rotation log + least squares.
    A_list, B_list are relative motions (4x4). Returns X (4x4).
    """
    assert len(A_list) == len(B_list) and len(A_list) >= 2
    N = len(A_list)

    M = np.zeros((3 * N, 3), dtype=np.float64)
    b = np.zeros((3 * N,), dtype=np.float64)

    for i in range(N):
        R_A = A_list[i][:3, :3]
        R_B = B_list[i][:3, :3]
        a = log_SO3(R_A)
        bb = log_SO3(R_B)
        M[3*i:3*i+3, :] = skew(a + bb)
        b[3*i:3*i+3] = bb - a

    rx = np.linalg.lstsq(M, b, rcond=None)[0]
    R_X = expm(skew(rx))

    C_blocks = []
    d_blocks = []
    for i in range(N):
        R_A = A_list[i][:3, :3]
        t_A = A_list[i][:3, 3]
        R_B = B_list[i][:3, :3]
        t_B = B_list[i][:3, 3]
        C_blocks.append(R_A - np.eye(3))
        d_blocks.append(R_X @ t_B - t_A)

    C = np.concatenate(C_blocks, axis=0)
    d = np.concatenate(d_blocks, axis=0)
    t_X = np.linalg.lstsq(C, d, rcond=None)[0]

    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R_X
    X[:3, 3] = t_X
    return X


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    Rm, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = tvec.reshape(3)
    return T


def T_to_list(T: np.ndarray) -> List[List[float]]:
    return T.astype(float).tolist()


# ===================== Rotation averaging (for base_T_board estimation) =====================
def avg_rotation(R_list: List[np.ndarray]) -> np.ndarray:
    M = np.zeros((3, 3), dtype=np.float64)
    for Rm in R_list:
        M += Rm
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


# ===================== Projection helpers =====================
def project_points(points_3d: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    imgpts, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    return imgpts.reshape(-1, 2).astype(int)


def draw_ee_axes(image: np.ndarray, cam_T_EE: np.ndarray, K: np.ndarray, dist: np.ndarray, axis_len: float = 0.10) -> np.ndarray:
    origin = cam_T_EE[:3, 3]
    x_axis = origin + axis_len * cam_T_EE[:3, 0]
    y_axis = origin + axis_len * cam_T_EE[:3, 1]
    z_axis = origin + axis_len * cam_T_EE[:3, 2]
    pts3 = np.stack([origin, x_axis, y_axis, z_axis], axis=0)
    pts2 = project_points(pts3, K, dist)
    o, x, y, z = [tuple(p) for p in pts2]
    cv2.line(image, o, x, (0, 0, 255), 3)
    cv2.line(image, o, y, (0, 255, 0), 3)
    cv2.line(image, o, z, (255, 0, 0), 3)
    return image


def draw_point(image: np.ndarray, uv: Tuple[int, int], color=(0, 0, 255), radius: int = 6) -> np.ndarray:
    cv2.circle(image, (int(uv[0]), int(uv[1])), radius, color, -1)
    return image


# ===================== Charuco helpers =====================
def create_charuco_board(board_size, square_len, marker_len, aruco_dict):
    # OpenCV 4.2: CharucoBoard_create
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            squaresX=board_size[0],
            squaresY=board_size[1],
            squareLength=square_len,
            markerLength=marker_len,
            dictionary=aruco_dict,
        )
    # Newer OpenCV: CharucoBoard(...) ctor exists
    return cv2.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)


def create_charuco_board_obj():
    return create_charuco_board(BOARD_SIZE, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)

def detect_charuco_board(image, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4.2-compatible detectMarkers call
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, ARUCO_DICT, parameters=ARUCO_PARAMS
    )

    if ids is None or len(corners) == 0:
        return None, None, None

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )

    return retval, charuco_corners, charuco_ids


# ===================== Data struct =====================
@dataclass
class Sample:
    base_T_EE: np.ndarray
    cam_T_board: np.ndarray


# ===================== Main calibrator =====================
class HandEyeCalibrator:
    def __init__(self, camera_name: str, setup: str, robot: XArmRobot):
        if camera_name not in CAMERA_TOPICS:
            raise RuntimeError(f"Unknown camera_name={camera_name}. Available: {list(CAMERA_TOPICS.keys())}")
        if setup not in ("eye_to_hand", "eye_in_hand"):
            raise RuntimeError("setup must be 'eye_to_hand' or 'eye_in_hand'")

        self.camera_name = camera_name
        self.setup = setup
        self.image_topic = CAMERA_TOPICS[camera_name]["image"]
        self.info_topic = CAMERA_TOPICS[camera_name]["info"]

        self.robot = robot
        self.bridge = CvBridge()
        self.board = create_charuco_board_obj()

        self.lock = threading.Lock()
        self.latest_image: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.frame_id: Optional[str] = None

        self.samples: List[Sample] = []

        # cached calibration
        self.base_T_cam: Optional[np.ndarray] = None      # eye_to_hand
        self.EE_T_cam: Optional[np.ndarray] = None        # eye_in_hand
        self.base_pose: Optional[np.ndarray] = None       # eye_in_hand_eval
        
        # keyboard requests
        self.capture_requested = False
        self.compute_requested = False
        self.eval_enabled = False
        self.reset_requested = False
        self.quit_requested = False

        rospy.Subscriber(self.image_topic, Image, self._on_image, queue_size=1)
        rospy.Subscriber(self.info_topic, CameraInfo, self._on_info, queue_size=1)

        keyboard.Listener(on_press=self._on_key).start()

        rospy.loginfo(f"[calib] camera_name={camera_name} setup={setup}")
        rospy.loginfo(f"[calib] image={self.image_topic}")
        rospy.loginfo(f"[calib] info ={self.info_topic}")
        rospy.loginfo("[calib] keys: s=capture  c=compute+save  e=eval  r=reset  q=quit")

    def _on_key(self, key):
        try:
            if key.char == 's':
                self.capture_requested = True
            elif key.char == 'c':
                self.compute_requested = True
            elif key.char == 'e':
                self.eval_enabled = not self.eval_enabled
                rospy.loginfo(f"[calib] eval_enabled={self.eval_enabled}")
            elif key.char == 'r':
                self.reset_requested = True
            elif key.char == 'q':
                self.quit_requested = True
        except AttributeError:
            pass

    def _on_info(self, msg: CameraInfo):
        with self.lock:
            self.width = msg.width
            self.height = msg.height
            self.frame_id = msg.header.frame_id
            self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self.dist = np.array(msg.D, dtype=np.float64).reshape(-1, 1)

    def _on_image(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self.lock:
            self.latest_image = img

    def _get_base_T_EE_from_robot(self) -> Optional[np.ndarray]:
        """
        Uses your ROS-based robot state, consistent with your TeleopDataCollector:
        st.ee_pose is expected to be [x(mm), y(mm), z(mm), roll(rad), pitch(rad), yaw(rad)]
        """
        st = self.robot.get_state()
        if st is None or getattr(st, "ee_pose", None) is None or len(st.ee_pose) < 6:
            return None
        ee = np.array(st.ee_pose[:6], dtype=np.float64)
        return euler_to_se3(ee, is_radian=True, pos_in_mm=True)

    def _reset(self):
        self.samples.clear()
        self.base_T_cam = None
        self.EE_T_cam = None
        rospy.loginfo("[calib] reset samples and cached calibration")

    def _try_capture(self):
        with self.lock:
            img = None if self.latest_image is None else self.latest_image.copy()
            K = None if self.K is None else self.K.copy()
            dist = None if self.dist is None else self.dist.copy()

        if img is None or K is None or dist is None:
            rospy.logwarn("[calib] waiting for image + CameraInfo ...")
            return

        retval, corners, ids = detect_charuco_board(img, self.board)
        if retval is None or retval < 5:
            rospy.logwarn("[calib] charuco not detected well enough")
            return

        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(corners, ids, self.board, K, dist, None, None)
        if not ok:
            rospy.logwarn("[calib] estimatePoseCharucoBoard failed")
            return

        base_T_EE = self._get_base_T_EE_from_robot()
        if base_T_EE is None:
            rospy.logwarn("[calib] robot ee_pose not available yet")
            return

        cam_T_board = rvec_tvec_to_T(rvec, tvec)
        self.samples.append(Sample(base_T_EE=base_T_EE, cam_T_board=cam_T_board))
        rospy.loginfo(f"[calib] captured sample #{len(self.samples)}")

    def _calib_eye_hand(self) -> np.ndarray:
        """
          - board fixed in table for in-hand calibration, board holded by robot gripper for to-hand calibration
        """
        R_base_gripper = []
        t_base_gripper = []
        R_cam_target = []
        t_cam_target = []

        for s in self.samples:
            t, R = s.base_T_EE[:3, 3], s.base_T_EE[:3, :3]
            if self.setup == "eye_to_hand":
                R = R.T
                t = -R.dot(t)
            R_base_gripper.append(R)
            t_base_gripper.append(t)
            R_cam_target.append(s.cam_T_board[:3, :3])
            t_cam_target.append(s.cam_T_board[:3, 3])

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_base_gripper,
            t_gripper2base=t_base_gripper,
            R_target2cam=R_cam_target,
            t_target2cam=t_cam_target,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        EE_T_cam = np.eye(4)
        EE_T_cam[:3, :3] = R_cam2gripper
        EE_T_cam[:3, 3] = t_cam2gripper.flatten()
        
        if self.setup == "eye_to_hand":
            base_T_cam = EE_T_cam
            EE_T_cam = None
        else:
            base_T_cam = None
            EE_T_cam = EE_T_cam
        
        return base_T_cam, EE_T_cam

    def _write_json(self, K: np.ndarray, dist: np.ndarray):
        out: Dict[str, Any] = {"version": 1, "cameras": {}}
        if os.path.exists(OUT_JSON):
            try:
                with open(OUT_JSON, "r", encoding="utf-8") as f:
                    out = json.load(f)
            except Exception:
                out = {"version": 1, "cameras": {}}
        if "cameras" not in out:
            out["cameras"] = {}

        out["cameras"][self.camera_name] = {
            "setup": self.setup,
            "updated_wall": time.time(),
            "samples": len(self.samples),
            "intrinsics": {
                "K": K.astype(float).tolist(),
                "dist": dist.reshape(-1).astype(float).tolist(),
                "width": int(self.width) if self.width is not None else None,
                "height": int(self.height) if self.height is not None else None,
                "source": "ROS CameraInfo",
            },
            "extrinsics": {
                "X_C": None if self.EE_T_cam is None else T_to_list(self.EE_T_cam),
                "frame": "RobotBase"
            } if self.setup == "eye_in_hand" else {
                "X_C": None if self.base_T_cam is None else T_to_list(self.base_T_cam),
                "frame": "EndEffector",
            }
        }

        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        rospy.loginfo(f"[calib] saved -> {OUT_JSON} (camera={self.camera_name}, setup={self.setup})")

    def _compute_and_save(self):
        if len(self.samples) < 5:
            rospy.logwarn("[calib] need >= 5 samples")
            return

        with self.lock:
            K = None if self.K is None else self.K.copy()
            dist = None if self.dist is None else self.dist.copy()

        if K is None or dist is None:
            rospy.logwarn("[calib] missing intrinsics (CameraInfo)")
            return

        if self.setup == "eye_in_hand":
            self.base_T_cam, self.EE_T_cam = self._calib_eye_hand()
            rospy.loginfo("[calib] computed EE_T_cam (eye_in_hand) and estimated base_T_board for eval")
        else:
            self.base_T_cam, self.EE_T_cam = self._calib_eye_hand()
            rospy.loginfo("[calib] computed base_T_cam (eye_to_hand)")

        self._write_json(K, dist)

    def _overlay_eval(self, img: np.ndarray, K: np.ndarray, dist: np.ndarray, corners, ids) -> np.ndarray:
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(corners, ids, self.board, K, dist, None, None)
        if not ok:
            return img

        # observed board axes
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.08)

        if self.setup == "eye_to_hand":
            if self.base_T_cam is None and len(self.samples) >= 5:
                self.base_T_cam, self.EE_T_cam = self._calib_eye_hand()
            if self.base_T_cam is None:
                return img

            base_T_EE = self._get_base_T_EE_from_robot()
            if base_T_EE is None:
                return img

            cam_T_EE = inv_T(self.base_T_cam) @ base_T_EE
            img = draw_ee_axes(img, cam_T_EE, K, dist, axis_len=0.10)
            return img

        # eye_in_hand (board fixed on table/base)
        if self.EE_T_cam is None and len(self.samples) >= 5:
            self.base_T_cam, self.EE_T_cam = self._calib_eye_hand()
        if self.EE_T_cam is None:
            return img

        base_T_EE = self._get_base_T_EE_from_robot()
        if base_T_EE is None:
            return img

        # Predict cam_T_base_pose:
        # cam_T_board_pred = inv(EE_T_cam) @ inv(base_T_EE) @ base_T_board
        self.base_pose = np.eye(4)
        cam_T_base_pose = inv_T(self.EE_T_cam) @ inv_T(base_T_EE) @ self.base_pose

        # draw predicted base origin as red point
        origin_cam = cam_T_base_pose[:3, 3].reshape(1, 3)
        uv = project_points(origin_cam, K, dist)[0]
        img = draw_point(img, (uv[0], uv[1]), color=(0, 0, 255), radius=6)
        
        return img

    def _show(self):
        with self.lock:
            img = None if self.latest_image is None else self.latest_image.copy()
            K = None if self.K is None else self.K.copy()
            dist = None if self.dist is None else self.dist.copy()

        if img is None or K is None or dist is None:
            return

        retval, corners, ids = detect_charuco_board(img, self.board)
        if retval is not None and corners is not None and ids is not None and retval >= 5:
            cv2.aruco.drawDetectedCornersCharuco(img, corners, ids)

        cv2.putText(img, f"cam={self.camera_name} setup={self.setup} samples={len(self.samples)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, "s=capture  c=compute+save  e=eval  r=reset  q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if self.eval_enabled and retval is not None and retval >= 5:
            img = self._overlay_eval(img, K, dist, corners, ids)

        cv2.imshow("handeye_charuco_ros1", img)
        cv2.waitKey(1)

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and not self.quit_requested:
            if self.reset_requested:
                self.reset_requested = False
                self._reset()

            if self.capture_requested:
                self.capture_requested = False
                self._try_capture()

            if self.compute_requested:
                self.compute_requested = False
                self._compute_and_save()

            self._show()
            rate.sleep()

        cv2.destroyAllWindows()


# ===================== main (auto-launch robot + selected camera) =====================
def main():
    rospy.init_node("handeye_charuco_calibration", anonymous=False)

    camera_name = rospy.get_param("~camera_name", "d405")
    setup = rospy.get_param("~setup", "eye_to_hand")  # eye_to_hand | eye_in_hand

    if camera_name not in CAMERA_TOPICS:
        rospy.logerr(f"[main] unknown camera_name={camera_name}, allowed={list(CAMERA_TOPICS.keys())}")
        raise SystemExit(1)
    if setup not in ("eye_to_hand", "eye_in_hand"):
        rospy.logerr("[main] ~setup must be 'eye_to_hand' or 'eye_in_hand'")
        raise SystemExit(1)

    teleop_cfg = TeleopConfig()
    services = XArmServices()

    procs: List[ManagedProcess] = []

    def shutdown():
        rospy.logwarn("[main] shutting down, stopping launched processes")
        for p in reversed(procs):
            try:
                p.stop()
            except Exception:
                pass

    rospy.on_shutdown(shutdown)

    # Auto-launch robot + selected realsense
    workdir = getattr(teleop_cfg, "launch_workdir", None) or os.getcwd()
    pipe_output = bool(getattr(teleop_cfg, "pipe_launch_output", False))

    # Robot bringup cmd from TeleopConfig (same style as your script)
    robot_cmd = list(getattr(teleop_cfg, "ROBOT_LAUNCH_CMD"))
    if robot_cmd:
        p = ManagedProcess("xarm_bringup", robot_cmd, workdir, pipe_output)
        p.start()
        procs.append(p)

    # Selected camera only
    cam_cmd = REALSENSE_CMDS.get(camera_name, None)
    if cam_cmd:
        p = ManagedProcess(f"realsense_{camera_name}", cam_cmd, workdir, pipe_output)
        p.start()
        procs.append(p)

    # Wait readiness
    rospy.loginfo("[startup] waiting for topics/services...")

    # Robot topic
    if not wait_for_topic(ROBOT_TOPIC, getattr(teleop_cfg, "startup_timeout_s", 20.0)):
        rospy.logerr(f"[startup] missing robot topic: {ROBOT_TOPIC}")
        raise SystemExit(1)

    # Robot services needed for manual mode
    must_srvs = [services.set_mode, services.set_state]
    missing = [s for s in must_srvs if not wait_for_service(s, getattr(teleop_cfg, "startup_timeout_s", 20.0))]
    if missing:
        rospy.logerr("[startup] missing services:\n  " + "\n  ".join(missing))
        raise SystemExit(1)

    # Camera topics
    image_t = CAMERA_TOPICS[camera_name]["image"]
    info_t = CAMERA_TOPICS[camera_name]["info"]
    if not wait_for_topic(image_t, getattr(teleop_cfg, "startup_timeout_s", 20.0)):
        rospy.logerr(f"[startup] missing camera image topic: {image_t}")
        raise SystemExit(1)
    if not wait_for_topic(info_t, getattr(teleop_cfg, "startup_timeout_s", 20.0)):
        rospy.logerr(f"[startup] missing camera info topic: {info_t}")
        raise SystemExit(1)

    rospy.loginfo("[startup] ready ✅")

    # Build robot interface (ROS-based)
    robot = XArmRobot(auto_init=False, debug=False)

    # Ensure manual mode (2) for calibration + eval
    try:
        robot.set_mode(MANUAL_MODE)
        robot.set_state(0)
        rospy.loginfo(f"[main] set robot manual mode={MANUAL_MODE}")
    except Exception as e:
        rospy.logwarn(f"[main] failed to set manual mode via robot wrapper: {e}")

    # Run calibrator
    calib = HandEyeCalibrator(camera_name=camera_name, setup=setup, robot=robot)
    calib.spin()


if __name__ == "__main__":
    main()