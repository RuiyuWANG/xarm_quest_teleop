# vr_pipeline/robot/xarm.py
import rospy
from dataclasses import dataclass
from typing import List, Optional, Tuple

from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import (
    Move, MoveRequest,
    GripperMove, GripperMoveRequest,
    SetInt16,
    GripperState,
    # velocity services are also in xarm_msgs on most installs
    VeloMove, VeloMoveRequest,   # <-- if this import fails, see note below
)

from .robot_config import (
    ROBOT_TOPIC,
    WAIT_FOR_FINISH_PARAM,
    GRIPPER_MIN, GRIPPER_MAX,
    HOME_JOINT,
    XArmServices,
)

@dataclass
class CallResult:
    ok: bool
    ret: int = 0
    message: str = ""

def _as_call_result(res) -> CallResult:
    # Most xarm services return fields ret + message
    ret = int(getattr(res, "ret", 0))
    msg = str(getattr(res, "message", ""))
    return CallResult(ok=(ret == 0), ret=ret, message=msg)

class XArmRobot:
    """
    Minimal xArm service-based wrapper matching your available services list.

    Ensures sequential execution by:
      - setting /xarm/wait_for_finish = True
      - checking service return codes before proceeding
    """

    def __init__(
        self,
        services: XArmServices = XArmServices(),
        home_joint: Optional[List[float]] = None,
        gripper_min: float = GRIPPER_MIN,
        gripper_max: float = GRIPPER_MAX,
        wait_for_finish: bool = True,
        auto_init: bool = True,
    ):
        self.services = services
        self.home_joint = HOME_JOINT if home_joint is None else home_joint
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max

        self._latest_robot_msg: Optional[RobotMsg] = None
        rospy.Subscriber(ROBOT_TOPIC, RobotMsg, self._robot_cb, queue_size=10)

        # force blocking behavior in xarm_ros services
        rospy.set_param(WAIT_FOR_FINISH_PARAM, bool(wait_for_finish))

        # Create proxies once
        self._set_mode = rospy.ServiceProxy(self.services.set_mode, SetInt16)
        self._set_state = rospy.ServiceProxy(self.services.set_state, SetInt16)

        self._go_home = rospy.ServiceProxy(self.services.go_home, SetInt16)  # go_home often uses SetInt16 (value ignored)

        self._move_joint = rospy.ServiceProxy(self.services.move_joint, Move)
        self._move_line = rospy.ServiceProxy(self.services.move_line, Move)

        self._velo_move_line_timed = rospy.ServiceProxy(self.services.velo_move_line_timed, VeloMove)

        self._gripper_move = rospy.ServiceProxy(self.services.gripper_move, GripperMove)
        self._gripper_state = rospy.ServiceProxy(self.services.gripper_state, GripperState)

        if auto_init:
            self.initialize()

    # ---------------- callbacks / state ----------------
    def _robot_cb(self, msg: RobotMsg):
        self._latest_robot_msg = msg

    def wait_for_state(self, timeout_s: float = 5.0) -> RobotMsg:
        start = rospy.Time.now().to_sec()
        r = rospy.Rate(200)
        while not rospy.is_shutdown():
            if self._latest_robot_msg is not None:
                return self._latest_robot_msg
            if rospy.Time.now().to_sec() - start > timeout_s:
                raise RuntimeError(f"Timeout waiting for {ROBOT_TOPIC}")
            r.sleep()
        raise RuntimeError("ROS shutdown")

    # ---------------- internal call helper ----------------
    def _call(self, srv_name: str, fn, req=None) -> CallResult:
        rospy.wait_for_service(srv_name)
        try:
            res = fn(req) if req is not None else fn()
            out = _as_call_result(res)
            if not out.ok:
                rospy.logwarn(f"[{srv_name}] failed ret={out.ret} msg={out.message}")
            return out
        except rospy.ServiceException as e:
            rospy.logerr(f"[{srv_name}] Service call failed: {e}")
            return CallResult(ok=False, ret=-1, message=str(e))

    # ---------------- 1) initialize robot & gripper ----------------
    def initialize(self) -> CallResult:
        """
        Initialize robot into a usable state.
        Equivalent to your pattern: set_mode(0), set_state(0), then home.
        """
        r1 = self.set_mode(0)
        if not r1.ok:
            return r1
        r2 = self.set_state(0)
        if not r2.ok:
            return r2
        # home robot + open gripper
        return self.home()

    # ---------------- 2) set mode/state ----------------
    def set_mode(self, mode: int) -> CallResult:
        return self._call(self.services.set_mode, self._set_mode, int(mode))

    def set_state(self, state: int) -> CallResult:
        return self._call(self.services.set_state, self._set_state, int(state))

    # ---------------- 3) gripper ----------------
    def get_gripper_state(self) -> Tuple[CallResult, float]:
        rospy.wait_for_service(self.services.gripper_state)
        try:
            res = self._gripper_state()
            out = _as_call_result(res)
            if not out.ok:
                return out, -1.0
            return out, float(res.curr_pos)
        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.services.gripper_state}] Service call failed: {e}")
            return CallResult(ok=False, ret=-1, message=str(e)), -1.0


    def move_gripper(self, pulse_pos: float) -> CallResult:
        assert isinstance(pulse_pos, (int, float)) and self.gripper_min <= pulse_pos <= self.gripper_max
        req = GripperMoveRequest()
        req.pulse_pos = float(pulse_pos)
        return self._call(self.services.gripper_move, self._gripper_move, req)

    # ---------------- 4) home ----------------
    def home(self) -> CallResult:
        """
        Home robot + open gripper.
        Uses /xarm/go_home then gripper_move.
        """
        # go_home signature varies; on many xarm_ros it's SetInt16 with no meaningful input
        r1 = self._call(self.services.go_home, self._go_home, 0)
        if not r1.ok:
            return r1
        return self.move_gripper(self.gripper_max)

    # ---------------- 5) move joint / move pose ----------------
    def move_to_joint(self, joints: List[float], mvvelo: float = 0, mvacc: float = 0, mvtime: float = 0) -> CallResult:
        assert isinstance(joints, list) and len(joints) == 7
        req = MoveRequest()
        req.pose = [float(x) for x in joints]
        req.mvvelo = float(mvvelo)
        req.mvacc = float(mvacc)
        req.mvtime = float(mvtime)
        return self._call(self.services.move_joint, self._move_joint, req)

    def move_to_pose(self, pose6: List[float], mvvelo: float = 0, mvacc: float = 0, mvtime: float = 0) -> CallResult:
        """
        pose6: [x_mm, y_mm, z_mm, roll, pitch, yaw] like your code uses
        """
        assert isinstance(pose6, list) and len(pose6) == 6
        req = MoveRequest()
        req.pose = [float(x) for x in pose6]
        req.mvvelo = float(mvvelo)
        req.mvacc = float(mvacc)
        req.mvtime = float(mvtime)
        return self._call(self.services.move_line, self._move_line, req)

    # ---------------- 6) tcp velocity control ----------------
    def velo_move_line_timed(self, velo: List[float], duration: float) -> CallResult:
        """
        Wrap /xarm/velo_move_line_timed.

        velo: [vx, vy, vz, wx, wy, wz] (units are whatever xarm expects; typically mm/s and rad/s, check srv)
        duration: seconds
        """
        assert isinstance(velo, list) and len(velo) == 6
        req = VeloMoveRequest()
        req.velo = [float(x) for x in velo]
        req.duration = float(duration)
        return self._call(self.services.velo_move_line_timed, self._velo_move_line_timed, req)
