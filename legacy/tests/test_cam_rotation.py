import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# robot and camera
from xarm.wrapper import XArmAPI
import palm.utils.transform_utils as TUtils
from img_utils import *

# ROBOT IP
# ROBOT_A_IP = "192.168.1.244"
ROBOT_B_IP = "192.168.1.243" # hold the camera

# Camera matrices
INTRINSICS_FILE = "charuco_intrinsics.npz"
HAND_EYE_FILE = "handeye_result.npz"
EYE_HAND_FILE = "eye_hand_result.npz"

handeye_data = np.load(HAND_EYE_FILE)
intr_data = np.load(INTRINSICS_FILE)
eyehand_data = np.load(EYE_HAND_FILE)

X_C = handeye_data["base_T_cam"]
K = intr_data["K"]
T = eyehand_data["base_T_cam"]

# Rotation in angles, in simulation the range is [-1 / 6 * np.pi, 1 / 6 * np.pi]
rot_angle = 1 / 12 * np.pi

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

def get_eef_pose(robot):
    code, pose = robot.get_position(is_radian=True)
    if code != 0:
        raise RuntimeError("Failed to get robot pose.")
    pose = np.array(pose)
    pose[:3] = pose[:3] / 1000  # Convert mm to m
    X = TUtils.Eular_to_SE3(pose)  # Convert to SE(3)
    return X

def SE3_to_6D_robot_action(X):
    eef_target = TUtils.SE3_to_Eular(X).flatten()
    eef_target[:3] = eef_target[:3] * 1000
    return eef_target

# robot_a = XArmAPI(ROBOT_A_IP)
robot_b = XArmAPI(ROBOT_B_IP)

robot_b.motion_enable(enable=True)
robot_b.set_mode(0)
robot_b.set_state(0)

X_C_p = get_new_cam_matrix_in_a(rot_angle, X_C)
curr_ee = get_eef_pose(robot_b)

eef_b_prime = get_new_cam_matrix_in_b(curr_ee, X_C, X_C_p, T)
eef_to_execute = SE3_to_6D_robot_action(eef_b_prime)
robot_b.set_position(x=eef_to_execute[0], y=eef_to_execute[1], z=eef_to_execute[2], roll=eef_to_execute[3], pitch=eef_to_execute[4], yaw=eef_to_execute[5], wait=True, is_radian=True)