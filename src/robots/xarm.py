import rospy
from dataclasses import dataclass
from typing import Sequence, Optional

from geometry_msgs.msg import PoseStamped, Twist

# Example: you already have these from xarm_ros
from xarm_msgs.srv import GripperMove, GripperMoveRequest

# You must replace these with the actual service types used by your xarm_ros for joint/pose/velocity/home.
# The wrappers are correct; just swap srv imports + request fields.
# from xarm_msgs.srv import MoveJoint, MoveJointRequest
# from xarm_msgs.srv import MoveLine, MoveLineRequest
# from xarm_msgs.srv import VcSetCartesianVelocity, VcSetCartesianVelocityRequest
# from xarm_msgs.srv import MoveHome, MoveHomeRequest
# from xarm_msgs.srv import Stop, StopRequest

@dataclass
class XArmServiceNames:
    gripper_move: str
    move_joint: str
    move_pose: str
    set_ee_twist: str
    stop: str
    home: str

class RobotXArm:
    def __init__(self, srv: XArmServiceNames, gripper_min=-100, gripper_max=850):
        self.srv = srv
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max

    # ---------- Gripper ----------
    def move_gripper(self, gripper_pos: float) -> int:
        assert isinstance(gripper_pos, (int, float))
        assert self.gripper_min <= gripper_pos <= self.gripper_max

        rospy.wait_for_service(self.srv.gripper_move)
        try:
            gripper_serv = rospy.ServiceProxy(self.srv.gripper_move, GripperMove)
            req = GripperMoveRequest()
            req.pulse_pos = float(gripper_pos)  # float32
            res = gripper_serv(req)
            if getattr(res, "ret", 0) != 0:
                rospy.logwarn(f"[gripper_move] Failed ret={res.ret} msg={getattr(res, 'message', '')}")
                return -1
            return 0
        except rospy.ServiceException as e:
            rospy.logerr(f"[gripper_move] Service call failed: {e}")
            return -1

    # ---------- Motion (replace srv types/fields with your xarm_ros) ----------
    def move_joints(self, joints_rad: Sequence[float], speed: float = 0.5, acc: float = 0.5) -> int:
        """
        joints_rad: list of joint angles in radians, length depends on arm DOF.
        speed/acc: normalized or physical depending on your xarm_ros API.
        """
        # rospy.wait_for_service(self.srv.move_joint)
        # try:
        #     serv = rospy.ServiceProxy(self.srv.move_joint, MoveJoint)
        #     req = MoveJointRequest()
        #     req.angles = list(map(float, joints_rad))
        #     req.speed = float(speed)
        #     req.acc = float(acc)
        #     res = serv(req)
        #     return 0 if res.ret == 0 else -1
        # except rospy.ServiceException as e:
        #     rospy.logerr(f"[move_joint] failed: {e}")
        #     return -1
        raise NotImplementedError("Wire this to your xarm_ros joint move service type/fields.")

    def move_pose(self, target: PoseStamped, speed: float = 0.2, acc: float = 0.2) -> int:
        """
        target: PoseStamped in robot base frame (or whatever your controller expects).
        """
        # rospy.wait_for_service(self.srv.move_pose)
        # try:
        #     serv = rospy.ServiceProxy(self.srv.move_pose, MoveLine)
        #     req = MoveLineRequest()
        #     req.pose = target  # or req.pose = target.pose depending on srv definition
        #     req.speed = float(speed)
        #     req.acc = float(acc)
        #     res = serv(req)
        #     return 0 if res.ret == 0 else -1
        # except rospy.ServiceException as e:
        #     rospy.logerr(f"[move_pose] failed: {e}")
        #     return -1
        raise NotImplementedError("Wire this to your xarm_ros pose move service type/fields.")

    def set_ee_twist(self, twist: Twist) -> int:
        """
        Cartesian velocity control for teleop.
        """
        # rospy.wait_for_service(self.srv.set_ee_twist)
        # try:
        #     serv = rospy.ServiceProxy(self.srv.set_ee_twist, VcSetCartesianVelocity)
        #     req = VcSetCartesianVelocityRequest()
        #     req.twist = twist  # or individual fields depending on srv definition
        #     res = serv(req)
        #     return 0 if res.ret == 0 else -1
        # except rospy.ServiceException as e:
        #     rospy.logerr(f"[set_ee_twist] failed: {e}")
        #     return -1
        raise NotImplementedError("Wire this to your xarm_ros cartesian velocity service type/fields.")

    def stop(self) -> int:
        # rospy.wait_for_service(self.srv.stop)
        # ...
        raise NotImplementedError("Wire this to your xarm_ros stop service.")

    def home(self) -> int:
        # rospy.wait_for_service(self.srv.home)
        # ...
        raise NotImplementedError("Wire this to your xarm_ros home service.")
