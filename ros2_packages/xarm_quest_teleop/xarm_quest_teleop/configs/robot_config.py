import numpy as np
from dataclasses import dataclass

# Topics
ROBOT_TOPIC = "/xarm/robot_states"
JOINT_STATES_TOPIC = "/xarm/joint_states"

WAIT_FOR_FINISH_PARAM = "/xarm/wait_for_finish"

# Services
SRV_SET_MODE = "/xarm/set_mode"
SRV_SET_STATE = "/xarm/set_state"
SRV_GO_HOME = "/xarm/go_home"
SRV_MOTION_ENABLE = "/xarm/motion_enable"
SRV_MOVE_JOINT = "/xarm/set_servo_angle"
SRV_MOVE_LINE = "/xarm/set_position"
SRV_MOVE_SERVO_CART = "/xarm/set_servo_cartesian"
SRV_VELO_MOVE_LINE_TIMED = "/xarm/vc_set_cartesian_velocity" # tcp velocity control

# Gripper
SRV_GRIPPER_MOVE = "/xarm/set_gripper_position"
SRV_GRIPPER_STATE = "/xarm/get_gripper_position"
SRV_SET_GRIPPER_MODE = "/xarm/set_gripper_mode"
SRV_SET_GRIPPER_ENABLE = "/xarm/set_gripper_enable"
SRV_SET_GRIPPER_SPEED = "/xarm/set_gripper_speed"

# Mode
MODE_POSITION = 0
MODE_EXT_TRAJ = 1
MODE_FREEDRIVE = 2
MODE_JOINT_VELO = 4
MODE_CART_VELO = 5
MODE_JOINT_ONLINE = 6
MODE_CART_ONLINE = 7
MODE_SERVO_CART = 1

# Gripper limits
GRIPPER_MIN = -10
GRIPPER_MAX = 850.0

# TCP Velocity limits
MAX_TCP_LIN_M_S = 0.25   # m/s
MAX_TCP_ANG_RAD_S = 1.5  # rad/s
ABS_SANITY_LIN_M_S = 2.0     # m/s
ABS_SANITY_ANG_RAD_S = 10.0  # rad/s

# Default home state
# HOME_JOINT = np.deg2rad([0, -45, 0, 45, 0, 88, 0]).tolist() # higher
HOME_JOINT = np.deg2rad([0, -60, -5, 21, -5, 80, 0]).tolist() # lower
HOME_GRIPPER = GRIPPER_MAX

@dataclass
class XArmServices:
    set_mode: str = SRV_SET_MODE
    set_state: str = SRV_SET_STATE
    motion_enable: str = SRV_MOTION_ENABLE
    go_home: str = SRV_GO_HOME
    move_joint: str = SRV_MOVE_JOINT
    move_line: str = SRV_MOVE_LINE
    velo_move_line_timed: str = SRV_VELO_MOVE_LINE_TIMED
    gripper_move: str = SRV_GRIPPER_MOVE
    gripper_state: str = SRV_GRIPPER_STATE
    set_gripper_mode: str = SRV_SET_GRIPPER_MODE
    set_gripper_enable: str = SRV_SET_GRIPPER_ENABLE
    set_gripper_speed: str = SRV_SET_GRIPPER_SPEED
    move_servo_cart: str = SRV_MOVE_SERVO_CART
