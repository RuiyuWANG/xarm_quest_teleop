# src/configs/quest2_config.py
"""
Quest2ROS API config (topics + default mapping).
"""

from dataclasses import dataclass

# ---------------- Topics ----------------
Q2R_RIGHT_HAND_POSE = "/q2r_right_hand_pose"
Q2R_RIGHT_HAND_TWIST = "/q2r_right_hand_twist"
Q2R_RIGHT_HAND_INPUTS = "/q2r_right_hand_inputs"
Q2R_RIGHT_HAND_HAPTIC = "/q2r_right_hand_haptic_feedback"

Q2R_LEFT_HAND_POSE = "/q2r_left_hand_pose"
Q2R_LEFT_HAND_TWIST = "/q2r_left_hand_twist"
Q2R_LEFT_HAND_INPUTS = "/q2r_left_hand_inputs"
Q2R_LEFT_HAND_HAPTIC = "/q2r_left_hand_haptic_feedback"


# ---------------- Default mapping ----------------
@dataclass
class Quest2Defaults:
    # Which hand controls robot by default
    active_hand: str = "right"  # "left" or "right"

    # Deadman enable (your demo uses button_lower)
    deadman_uses_lower_button: bool = True

    # Gripper mapping (your demo uses index/middle triggers)
    # Convention we’ll use in teleop:
    #   press_index  -> gripper close
    #   press_middle -> gripper open (or haptic amp)
    use_press_index_for_grip: bool = True
