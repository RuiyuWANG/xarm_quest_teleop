import os
import time
import numpy as np
import cv2

# import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

INTRINSICS_FILE = "charuco_intrinsics.npz"
HAND_EYE_FILE = "handeye_result.npz"
handeye_data = np.load(HAND_EYE_FILE)
X_C = handeye_data["base_T_cam"]
intr_data = np.load(INTRINSICS_FILE)
K = intr_data["K"]

# def setup_camera():
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#     pipeline.start(config)
#     sensor = pipeline.get_active_profile().get_device().first_color_sensor()
#     sensor.set_option(rs.option.enable_auto_white_balance, 0)
#     sensor.set_option(rs.option.white_balance, 3400)
#     return pipeline


def get_object_mask(image, id, task="stack_two"):
    assert task == "stack_two", "Only 'stack_two' task is supported"
    assert id in [0, 1], "Object ID must be 0 (blue) or 1 (yellow)"

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, W = hsv.shape[:2]

    # Define HSV color bounds (based on your adjusted values)
    if id == 0:  # Blue cube
        lower = np.array([90, 60, 100])
        upper = np.array([130, 255, 255])

        # Step 1: Color mask
        object_mask = cv2.inRange(hsv, lower, upper)
    elif id == 1:  # red cube
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])

        lower2 = np.array([170, 100, 100])
        upper2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        object_mask = cv2.bitwise_or(mask1, mask2)

    # Step 2: Create center region mask
    cx, cy = W // 2, H * 2 // 3
    region_w, region_h = W // 2, H // 2
    center_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(
        center_mask,
        (cx - region_w // 2, cy - region_h // 2),
        (cx + region_w // 2, cy + region_h // 2),
        255,
        -1,
    )

    # Step 3: Apply center mask first
    center_filtered = cv2.bitwise_and(object_mask, center_mask)

    # Step 4: Find the largest contour in the filtered region
    contours, _ = cv2.findContours(center_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_mask = np.zeros_like(object_mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(largest_mask, [largest], -1, 255, -1)

    return largest_mask


def cam_pt_to_table_pt(cam_pt, table_height):
    R, t = X_C[:3, :3], X_C[:3, 3]

    p_image = np.array([cam_pt[0], cam_pt[1], 1])
    p_camera = np.linalg.inv(K) @ p_image
    p_world_ray = R @ p_camera

    s = (table_height - t[2]) / p_world_ray[2]
    p_world = t + s * p_world_ray
    return p_world


def project_points(points_3d, K):
    """
    Project 3D points to 2D using intrinsic matrix (no distortion support).

    Args:
        points_3d: (N, 3) numpy array of 3D points in camera frame
        K: (3, 3) intrinsic matrix

    Returns:
        (N, 2) pixel coordinates
    """
    points_3d = np.asarray(points_3d)
    assert points_3d.shape[1] == 3

    # Apply pinhole projection
    points_2d = (K @ points_3d.T).T  # shape (N, 3)
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    return points_2d.astype(int)


def project_ee_axes(camera_T_EE, K, axis_length=0.1):
    """
    Projects the EE frame axes to pixel space.

    Args:
        camera_T_EE: (4, 4) EE pose in camera frame
        K: (3, 3) camera intrinsic matrix
        axis_length: length of the axis in meters

    Returns:
        pixel_points: (4, 2) pixel coordinates of origin, x, y, z axes
    """
    origin = camera_T_EE[:3, 3]

    # EE axes in its local frame
    x_axis = origin + axis_length * camera_T_EE[:3, 0]
    y_axis = origin + axis_length * camera_T_EE[:3, 1]
    z_axis = origin + axis_length * camera_T_EE[:3, 2]

    pts_3d = np.stack([origin, x_axis, y_axis, z_axis], axis=0)  # shape (4, 3)
    pts_2d = project_points(pts_3d, K)  # shape (4, 2)
    return pts_2d  # pixel coordinates


def get_object_position_from_mask(mask_gray):
    # Get non-zero (white) pixel positions
    object_coords = np.column_stack(np.nonzero(mask_gray))  # shape: (N, 2), [row, col]

    if object_coords.size == 0:
        print("Occlusion or missing object — unable to get object mask!")
        return None

    object_centroid = object_coords.mean(axis=0)  # (row, col)
    y, x = int(object_centroid[0]), int(object_centroid[1])  # row = y, col = x

    return np.array([x, y])


def sample_pose_within_workspace(x0, y0, x_bounds, y_bounds, slack):
    x = np.random.uniform(0, x_bounds - slack * 2) + x0 + slack
    y = np.random.uniform(0, y_bounds - slack * 2) + y0 + slack
    return x, y


def sample_ee_xy_rotation(R_curr: np.ndarray, angle_range_deg=(-45, 45)) -> np.ndarray:
    """
    Given current EE rotation matrix R_curr (3x3),
    return a new rotation matrix R_new that rotates around EE's local Z axis
    by a random angle in degrees within angle_range_deg.

    Args:
        R_curr: np.ndarray, shape (3, 3), current EE rotation matrix
        angle_range_deg: tuple, min and max rotation in degrees

    Returns:
        R_new: np.ndarray, shape (3, 3), new EE rotation matrix
    """
    assert R_curr.shape == (3, 3), "R_curr must be a 3x3 rotation matrix."

    # Sample angle in radians
    angle_deg = np.random.uniform(*angle_range_deg)
    angle_rad = np.deg2rad(angle_deg)

    # Rotation around local Z (EE's Z axis)
    R_delta = R.from_euler("z", angle_rad).as_matrix()

    # Apply delta rotation in EE's local frame
    R_new = R_curr @ R_delta
    return R_new


def project_obj_pose_to_img(curr_ee_pose, obj_position, K, X_C):
    """
    Projects the rotated EE axes to image space.

    Args:
        curr_ee_pose: np.ndarray, shape (4,4), current SE(3) of EE (world_T_EE)
        img: np.ndarray, image (not used in projection, but may be used for drawing)
        obj_position: list or np.ndarray of shape (3,), new EE translation
        K: np.ndarray, shape (3,3), camera intrinsics
        X_C: np.ndarray, shape (4,4), camera pose in world (world_T_camera)

    Returns:
        imgpts: list of 2D projected points for EE axes in image space
    """
    assert curr_ee_pose.shape == (4, 4), "curr_ee_pose must be SE(3)"
    assert X_C.shape == (4, 4), "X_C must be SE(3)"
    assert len(obj_position) == 3

    # Create new EE pose with a rotation around its own Z
    new_ee_pose = np.eye(4)
    new_ee_pose[:3, :3] = sample_ee_xy_rotation(curr_ee_pose[:3, :3])
    new_ee_pose[:3, 3] = obj_position

    # Compute EE pose relative to camera (camera_T_EE)
    camera_T_EE = np.linalg.inv(X_C) @ new_ee_pose

    # Project axes
    imgpts = project_ee_axes(camera_T_EE, K)
    return imgpts


def get_new_cam_matrix_in_a(rad, X_C):
    X_M = np.eye(4)
    M_X_C = np.linalg.inv(X_M) @ X_C

    # Rotate the camera around the movement center
    delta_X = np.eye(4)
    delta_X[:3, :3] = R.from_rotvec(rad * np.array([0, 0, 1])).as_matrix()
    M_X_C = delta_X @ M_X_C

    return X_M @ M_X_C


def get_new_cam_matrix_in_b(curr_ee, X_C, X_C_prime, T):
    ee_prime = curr_ee @ T @ np.linalg.inv(X_C) @ X_C_prime @ np.linalg.inv(T)
    
    return ee_prime
