from dataclasses import dataclass

#  Topics 
Q2R_RIGHT_HAND_POSE = "/q2r_right_hand_pose"
Q2R_RIGHT_HAND_TWIST = "/q2r_right_hand_twist"
Q2R_RIGHT_HAND_INPUTS = "/q2r_right_hand_inputs"
Q2R_RIGHT_HAND_HAPTIC = "/q2r_right_hand_haptic_feedback"

Q2R_LEFT_HAND_POSE = "/q2r_left_hand_pose"
Q2R_LEFT_HAND_TWIST = "/q2r_left_hand_twist"
Q2R_LEFT_HAND_INPUTS = "/q2r_left_hand_inputs"
Q2R_LEFT_HAND_HAPTIC = "/q2r_left_hand_haptic_feedback"

@dataclass
class Quest2Defaults:
    active_hand: str = "right"  # "left" or "right"

    # Deadman enable
    deadman_uses_lower_button: bool = True

    # Gripper mapping
    #   press_index  -> gripper close
    #   press_middle -> gripper open
    use_press_index_for_grip: bool = True
