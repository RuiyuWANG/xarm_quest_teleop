import numpy as np
from geometry_msgs.msg import Twist

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class VRTeleopController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.deadman_field = cfg["deadman_button"]          # "button_upper"
        self.gripper_field = cfg["gripper_axis"]            # "press_index"
        self.p_open = float(cfg["gripper_pulse_open"])
        self.p_close = float(cfg["gripper_pulse_close"])
        self.max_lin = float(cfg.get("max_ee_linear", 0.25))
        self.max_ang = float(cfg.get("max_ee_angular", 1.5))

    def enabled(self, inputs_msg) -> bool:
        if inputs_msg is None:
            return False
        return bool(getattr(inputs_msg, self.deadman_field))

    def gripper_pulse(self, inputs_msg) -> float:
        if inputs_msg is None:
            return self.p_open
        a = float(getattr(inputs_msg, self.gripper_field))
        a = clamp(a, 0.0, 1.0)
        return self.p_open + a * (self.p_close - self.p_open)

    def ee_twist_cmd(self, twist_msg) -> Twist:
        """
        Scale/clamp twist into safe bounds.
        """
        out = Twist()
        if twist_msg is None:
            return out

        out.linear.x = float(np.clip(twist_msg.linear.x, -self.max_lin, self.max_lin))
        out.linear.y = float(np.clip(twist_msg.linear.y, -self.max_lin, self.max_lin))
        out.linear.z = float(np.clip(twist_msg.linear.z, -self.max_lin, self.max_lin))

        out.angular.x = float(np.clip(twist_msg.angular.x, -self.max_ang, self.max_ang))
        out.angular.y = float(np.clip(twist_msg.angular.y, -self.max_ang, self.max_ang))
        out.angular.z = float(np.clip(twist_msg.angular.z, -self.max_ang, self.max_ang))
        return out
