# vr_pipeline/robot/robot_config.py
import numpy as np
from dataclasses import dataclass

# Topics
ROBOT_TOPIC = "/xarm/xarm_states"
JOINT_STATES_TOPIC = "/xarm/joint_states"

# Common ROS param used by xarm_ros driver: if True, services block until motion finishes
WAIT_FOR_FINISH_PARAM = "/xarm/wait_for_finish"

# Services
SRV_SET_MODE = "/xarm/set_mode"
SRV_SET_STATE = "/xarm/set_state"
SRV_GO_HOME = "/xarm/go_home"

SRV_MOVE_JOINT = "/xarm/move_joint"
SRV_MOVE_LINE = "/xarm/move_line"

# TCP velocity control
SRV_VELO_MOVE_LINE_TIMED = "/xarm/velo_move_line_timed"

# Gripper
SRV_GRIPPER_MOVE = "/xarm/gripper_move"
SRV_GRIPPER_STATE = "/xarm/gripper_state"

# Error
SRV_CLEAR_ERR = "/xarm/clear_err"

# Mode
MODE_POSITION = 0
MODE_EXT_TRAJ = 1
MODE_FREEDRIVE = 2
MODE_JOINT_VELO = 4
MODE_CART_VELO = 5
MODE_JOINT_ONLINE = 6
MODE_CART_ONLINE = 7

# Default limits
GRIPPER_MIN = -10
GRIPPER_MAX = 850

# TCP Velocity limits
MAX_TCP_LIN_M_S = 0.25   # m/s
MAX_TCP_ANG_RAD_S = 1.5  # rad/s
ABS_SANITY_LIN_M_S = 2.0     # m/s
ABS_SANITY_ANG_RAD_S = 10.0  # rad/s

# Default home joint (adjust to your arm; keep same as your example)
HOME_JOINT = np.deg2rad([0, -45, 0, 45, 0, 88, 0]).tolist()
HOME_GRIPPER = GRIPPER_MAX

@dataclass
class XArmServices:
    set_mode: str = SRV_SET_MODE
    set_state: str = SRV_SET_STATE
    go_home: str = SRV_GO_HOME
    move_joint: str = SRV_MOVE_JOINT
    move_line: str = SRV_MOVE_LINE
    velo_move_line_timed: str = SRV_VELO_MOVE_LINE_TIMED
    gripper_move: str = SRV_GRIPPER_MOVE
    gripper_state: str = SRV_GRIPPER_STATE
    move_servo_cart: str = "/xarm/move_servo_cart"
    motion_ctrl: str = "/xarm/motion_ctrl"

