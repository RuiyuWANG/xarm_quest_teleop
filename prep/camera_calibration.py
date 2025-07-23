import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import palm.utils.transform_utils as TUtils
from pynput import keyboard
import os
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm


# ======================== CONFIGURATION ========================
BOARD_SIZE = (7, 5)           # Number of squares (width, height)
SQUARE_LENGTH = 0.035         # Meters
MARKER_LENGTH = 0.026          # Meters
EYE_TO_HAND = True

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
INTRINSICS_FILE = "charuco_intrinsics.npz"
HAND_EYE_FILE = "handeye_result.npz"
EYE_HAND_FILE = "eye_hand_result.npz"
CALIBRATION_DATA_FILE = "hand_eye_calibration_data.npz"
EYE_HAND_DATA_FILE = "eye_hand_calibration_data.npz"
ROBOT_IP = "192.168.1.244"

# ======================== CALIBRATION FUNCTION S ========================
def log_SO3(R):
    """Logarithm of a rotation matrix."""
    theta = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(theta, 0):
        return np.zeros((3,))
    lnR = theta / (2 * np.sin(theta)) * (R - R.T)
    return np.array([lnR[2, 1], lnR[0, 2], lnR[1, 0]])

def skew(v):
    """Return skew-symmetric matrix of vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def solve_AX_XB(A_list, B_list):
    """
    Solves AX = XB for rotation and translation separately.

    Args:
        A_list: list of 4x4 numpy arrays (robot motion)
        B_list: list of 4x4 numpy arrays (camera target motion)

    Returns:
        X: 4x4 transformation matrix from camera to gripper
    """
    assert len(A_list) == len(B_list)
    N = len(A_list)

    # === Solve for rotation: R_A * R_X = R_X * R_B
    M = np.zeros((3 * N, 3))
    b = np.zeros((3 * N,))
    for i in range(N):
        R_A = A_list[i][:3, :3]
        R_B = B_list[i][:3, :3]
        log_R_A = log_SO3(R_A)
        log_R_B = log_SO3(R_B)
        M[3*i:3*i+3, :] = skew(log_R_A + log_R_B)
        b[3*i:3*i+3] = log_R_B - log_R_A

    # Least-squares solve
    rx = np.linalg.lstsq(M, b, rcond=None)[0]
    R_X = expm(skew(rx))

    # === Solve for translation: R_A * t_X + t_A = R_X * t_B + t_X
    C = []
    d = []
    for i in range(N):
        R_A = A_list[i][:3, :3]
        t_A = A_list[i][:3, 3]
        R_B = B_list[i][:3, :3]
        t_B = B_list[i][:3, 3]
        C.append(R_A - np.eye(3))
        d.append(R_X @ t_B - t_A)

    C = np.concatenate(C, axis=0)
    d = np.concatenate(d, axis=0)
    t_X = np.linalg.lstsq(C, d, rcond=None)[0]

    # Combine into full SE3
    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X
    return X

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


def create_charuco_board():
    """Create Charuco board object"""
    return cv2.aruco.CharucoBoard(BOARD_SIZE, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)

def detect_charuco_board(image, board):
    """Detect Charuco board in an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
    
    if len(corners) == 0:
        return None, None, None
    
    # Interpolate Charuco corners
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    
    return retval, charuco_corners, charuco_ids

def calibrate_camera_intrinsics(pipeline, num_samples=20):
    """Calibrate camera intrinsics using Charuco board"""
    global capture_requested, quit_requested
    board = create_charuco_board()
    all_corners = []
    all_ids = []
    sample_count = 0
    
    print(f"\n=== CAMERA INTRINSIC CALIBRATION ===\n")
    print(f"Collecting {num_samples} samples...")
    print("Move board around to cover different orientations and distances")
    print("Press 's' to capture, 'q' to finish early")
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    try:
        while sample_count < num_samples and not quit_requested:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect board
            retval, corners, ids = detect_charuco_board(image, board)
            vis_image = image.copy()
            
            if retval and retval > 4:
                cv2.aruco.drawDetectedCornersCharuco(vis_image, corners, ids)
            
            cv2.putText(vis_image, f"Samples: {sample_count}/{num_samples}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_image, "Press 's' to capture", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Intrinsic Calibration", vis_image)
            cv2.waitKey(1)
            
            if capture_requested:
                capture_requested = False
                if retval and retval > 4:
                    all_corners.append(corners)
                    all_ids.append(ids)
                    sample_count += 1
                    print(f"Captured sample {sample_count}/{num_samples}")
                else:
                    print("Board not detected properly. Try again.")
    
    finally:
        cv2.destroyAllWindows()
        listener.stop()
    
    if len(all_corners) < 5:
        print("Not enough samples for calibration (min 5 required)")
        return None, None
    
    print("Calibrating camera...")
    img_size = gray.shape[::-1]
    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None
    )
    
    if ret:
        print("Calibration successful!")
        print(f"Reprojection error: {ret:.4f}")
        print("Camera Matrix:\n", K)
        print("Distortion Coefficients:\n", dist)
        
        # Save results
        np.savez(INTRINSICS_FILE, K=K, dist=dist, img_size=img_size)
        print(f"Saved intrinsics to {INTRINSICS_FILE}")
        return K, dist
    
    print("Calibration failed")
    return None, None

def calibrate_hand_eye(arm, pipeline, K, dist, num_samples=20, eye_to_hand=True):
    global capture_requested, quit_requested
    """Perform hand-eye calibration (eye-to-hand setup)"""
    board = create_charuco_board()
    base_T_EE_list = []
    camera_T_board_list = []
    sample_count = 0
    
    data_file = CALIBRATION_DATA_FILE if eye_to_hand else EYE_HAND_DATA_FILE
    if os.path.exists(data_file):
        print(f"Loading existing calibration data from: {data_file}")
        data = np.load(data_file, allow_pickle=True)
        base_T_EE_list = list(data["base_T_EE_list"])
        camera_T_board_list = list(data["camera_T_board_list"])

    else:
        
        print(f"\n=== HAND-EYE CALIBRATION ===\n")
        print("Robot will hold the calibration board")
        print(f"Collecting {num_samples} samples...")
        print("Move robot to different poses, covering workspace")
        print("Press 's' to capture, 'q' to finish early")
        
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        
        try:
            while sample_count < num_samples and not quit_requested:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                image = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect board
                retval, corners, ids = detect_charuco_board(image, board)
                vis_image = image.copy()
                
                if retval and retval > 4:
                    cv2.aruco.drawDetectedCornersCharuco(vis_image, corners, ids)
                    
                    # Estimate board pose
                    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        corners, ids, board, K, dist, None, None
                    )
                    if success:
                        # Draw axis
                        cv2.drawFrameAxes(vis_image, K, dist, rvec, tvec, 0.1)
                
                cv2.putText(vis_image, f"Samples: {sample_count}/{num_samples}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_image, "Press 's' to capture", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Hand-Eye Calibration", vis_image)
                cv2.waitKey(1)
                
                if capture_requested:
                    capture_requested = False
                    if retval and retval > 4 and success:
                        # Get robot pose
                        code, pose = arm.get_position(is_radian=True)
                        if code != 0:
                            print(f"Failed to get robot pose: {code}")
                            continue
                        
                        pose = np.array(pose)
                        pose[:3] = pose[:3] / 1000
                        base_T_EE = TUtils.Eular_to_SE3(pose)
                        TUtils.check_SE3(base_T_EE)
                        
                        # Create camera to board transform
                        R_board, _ = cv2.Rodrigues(rvec)
                        camera_T_board = np.eye(4)
                        camera_T_board[:3, :3] = R_board
                        camera_T_board[:3, 3] = tvec.flatten()
                        
                        base_T_EE_list.append(base_T_EE)
                        camera_T_board_list.append(camera_T_board)
                        sample_count += 1
                        print(f"Captured sample {sample_count}/{num_samples}")

                    else:
                        print("Board not detected or pose estimation failed. Try again.")
        
        finally:
            cv2.destroyAllWindows()
            listener.stop()
        
        if len(base_T_EE_list) < 5:
            print("Not enough samples for calibration (min 5 required)")
            return None
        
        # Save collected data for future runs
        data_file = CALIBRATION_DATA_FILE if eye_to_hand else EYE_HAND_DATA_FILE
        np.savez(data_file,
                base_T_EE_list=base_T_EE_list,
                camera_T_board_list=camera_T_board_list)
        print(f"Saved calibration samples to {data_file}")
    
    # Prepare data for hand-eye calibration
    R_base_gripper = []
    t_base_gripper = []
    R_cam_target = []
    t_cam_target = []
    
    for base_T_EE, camera_T_board in zip(base_T_EE_list, camera_T_board_list):
        t, R = base_T_EE[:3, 3], base_T_EE[:3, :3]
        if eye_to_hand:
            R = R.T
            t = -R.dot(t)
        R_base_gripper.append(R)
        t_base_gripper.append(t)
        R_cam_target.append(camera_T_board[:3, :3])
        t_cam_target.append(camera_T_board[:3, 3])
    
    # Perform hand-eye calibration (eye-to-hand setup)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_base_gripper,
        t_gripper2base=t_base_gripper,
        R_target2cam=R_cam_target,
        t_target2cam=t_cam_target,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # Form base to camera transform
    EE_T_cam = np.eye(4)
    EE_T_cam[:3, :3] = R_cam2gripper
    EE_T_cam[:3, 3] = t_cam2gripper.flatten()
    
    if eye_to_hand:
        base_T_cam = EE_T_cam
        EE_T_cam = np.eye(4)
    else:
        base_T_cam = base_T_EE_list[0] @ EE_T_cam
    
    print("\n=== CALIBRATION RESULTS ===")
    print("Base to Camera Transform:\n", base_T_cam)
    print("\nEE to Camera Transform:\n", EE_T_cam)
    
    # Save results
    save_file = HAND_EYE_FILE if eye_to_hand else EYE_HAND_FILE
    np.savez(save_file,
             base_T_cam=base_T_cam, 
             EE_T_cam=EE_T_cam,
             base_T_EE_list=base_T_EE_list,
             camera_T_board_list=camera_T_board_list,
             K=K,
             dist=dist)
    print(f"Saved hand-eye calibration to {save_file}")
    
    return base_T_cam

def visualize_calibration(pipeline, K, dist, base_T_cam):
    """Visualize calibration results by projecting EE frame"""
    arm = XArmAPI(ROBOT_IP)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    
    print("\n=== VISUALIZATION ===\n")
    print("Projecting EE coordinate frame on camera image")
    print("Press 'q' to exit")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            
            # Get robot pose
            code, pose = arm.get_position(is_radian=True)
            if code != 0:
                continue
            
            pose = np.array(pose)

            pose[:3] = pose[:3] / 1000
            base_T_EE = TUtils.Eular_to_SE3(pose)
            
            # Compute camera to EE transform: camera_T_EE = inv(base_T_cam) @ base_T_EE
            camera_T_EE = np.linalg.inv(base_T_cam) @ base_T_EE
            imgpts = project_ee_axes(camera_T_EE, K)
            
            # # Draw axese
            cv2.line(image, tuple(imgpts[0]), tuple(imgpts[1]), (0,0,255), 3)  # X - Red
            cv2.line(image, tuple(imgpts[0]), tuple(imgpts[2]), (0,255,0), 3)  # Y - Green
            cv2.line(image, tuple(imgpts[0]), tuple(imgpts[3]), (255,0,0), 3)  # Z - Blue
            
            cv2.imshow("Calibration Visualization", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        arm.disconnect()

# ======================== GLOBAL STATE ========================
capture_requested = False
quit_requested = False
    
def on_press(key):
    """Keyboard listener callback"""
    global capture_requested, quit_requested
    try:
        if key.char == 's':
            capture_requested = True
        elif key.char == 'q':
            quit_requested = True
    except AttributeError:
        pass

# ======================== MAIN EXECUTION ========================
if __name__ == "__main__":
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    
    # Initialize robot
    arm = XArmAPI(ROBOT_IP)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    
    try:
        # Step 1: Camera Intrinsic Calibration
        if not os.path.exists(INTRINSICS_FILE):
            print("Performing camera intrinsic calibration...")
            K, dist = calibrate_camera_intrinsics(pipeline, num_samples=30)
            if K is None:
                print("Failed to calibrate camera intrinsics")
                exit(1)
        else:
            print("Loading existing intrinsics...")
            intr_data = np.load(INTRINSICS_FILE)
            K = intr_data["K"]
            dist = intr_data["dist"]
        
        # Reset flags for next calibration
        capture_requested = False
        quit_requested = False
        
        # Step 2: Hand-Eye Calibration
        
        save_file = HAND_EYE_FILE if EYE_TO_HAND else EYE_HAND_FILE
        if not os.path.exists(save_file):
            print("\nPerforming hand-eye calibration...")
            base_T_cam = calibrate_hand_eye(arm, pipeline, K, dist, num_samples=20, eye_to_hand=EYE_TO_HAND)
            if base_T_cam is None:
                print("Failed to perform hand-eye calibration")
                exit(1)
        else:
            print("Loading existing hand-eye calibration...")
            handeye_data = np.load(save_file)
            base_T_cam = handeye_data["base_T_cam"]
        
        # Step 3: Visualization
        print("\nStarting visualization...")
        print(K, base_T_cam)
        visualize_calibration(pipeline, K, dist, base_T_cam)
        
    finally:
        pipeline.stop()
        arm.disconnect()
        print("Cleanup complete")