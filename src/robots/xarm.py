import signal
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rospy

from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import (
    Move, MoveRequest,
    GripperMove, GripperMoveRequest,
    SetInt16, SetInt16Request,
    GripperState,
    MoveVelocity, MoveVelocityRequest,
)

from src.configs.robot_config import (
    ROBOT_TOPIC,
    WAIT_FOR_FINISH_PARAM,
    GRIPPER_MIN, GRIPPER_MAX,
    HOME_JOINT, HOME_GRIPPER,
    MODE_CART_VELO, MODE_POSITION, MODE_SERVO_CART, 
    MAX_TCP_LIN_M_S, MAX_TCP_ANG_RAD_S,
    ABS_SANITY_LIN_M_S, ABS_SANITY_ANG_RAD_S,
    XArmServices,
)
from src.utils.robot_utils import speeds6_mps_to_xarm_units, _wrap_to_pi


@dataclass
class CallResult:
    ok: bool
    ret: int = 0
    message: str = ""


@dataclass
class XArmState:
    ee_pose: Optional[List[float]]
    joint_angles: Optional[List[float]]
    gripper_qpos: Optional[float]
    mode: Optional[int]
    state: Optional[int]
    err: Optional[int]
    warn: Optional[int]

def _as_call_result(res) -> CallResult:
    ret = int(getattr(res, "ret", 0))
    msg = str(getattr(res, "message", ""))
    return CallResult(ok=(ret == 0), ret=ret, message=msg)


class XArmRobot:
    """
    ROS1 xArm wrapper that provides ROS2-like "spin till done" semantics WITHOUT relying on cmdnum.

    Completion logic:
      - For position control (move_joint/move_line):
          wait until current state is within tolerance of the goal (joint/pose compare).
      - For timed Cartesian velocity control:
          sleep for duration (+ small margin), optionally wait until idle.

    Also:
      - Ctrl+C clean exit: breaks all wait loops and (optionally) calls /xarm/motion_ctrl if configured.

    RobotMsg fields you confirmed:
      int16 state, mode, cmdnum, err, warn
      float32[] angle
      float32[6] pose
    """

    def __init__(
        self,
        services: XArmServices = XArmServices(),
        wait_for_finish_param: bool = True,
        auto_init: bool = True,
        debug: bool = True,
        # default tolerances
        joint_tol_rad: float = 0.01,   # ~0.57 deg
        pose_tol_mm: float = 1.0,
        pose_tol_rad: float = 0.03,    # ~1.7 deg
    ):
        self.services = services
        self.debug = debug

        self.joint_tol_rad = float(joint_tol_rad)
        self.pose_tol_mm = float(pose_tol_mm)
        self.pose_tol_rad = float(pose_tol_rad)

        self._latest_robot_msg: Optional[RobotMsg] = None

        self._stop_requested: bool = False
        signal.signal(signal.SIGINT, self._on_sigint)

        rospy.Subscriber(ROBOT_TOPIC, RobotMsg, self._robot_cb, queue_size=10)

        # Keep setting this param, but DO NOT rely on it for blocking.
        rospy.set_param(WAIT_FOR_FINISH_PARAM, bool(wait_for_finish_param))

        # Service proxies
        self._set_mode = rospy.ServiceProxy(self.services.set_mode, SetInt16)
        self._set_state = rospy.ServiceProxy(self.services.set_state, SetInt16)

        self._move_joint = rospy.ServiceProxy(self.services.move_joint, Move)
        self._move_line = rospy.ServiceProxy(self.services.move_line, Move)

        self._velo_move_line_timed = rospy.ServiceProxy(self.services.velo_move_line_timed, MoveVelocity)

        self._gripper_move = rospy.ServiceProxy(self.services.gripper_move, GripperMove)
        self._gripper_state = rospy.ServiceProxy(self.services.gripper_state, GripperState)
        self._move_servo_cart = rospy.ServiceProxy(self.services.move_servo_cart, Move)

        self.wait_for_state(timeout_s=10.0)

        if auto_init:
            r = self.initialize()
            if not r.ok:
                raise RuntimeError(f"initialize failed: ret={r.ret} msg={r.message}")
            
        rospy.loginfo("[XArmRobot] initialized")

    # ---------------- signal handling ----------------
    def _on_sigint(self, signum, frame):
        if self._stop_requested:
            return
        self._stop_requested = True
        rospy.logwarn("[XArmRobot] Ctrl+C (SIGINT) received. Stopping waits and requesting shutdown...")
        # try:
        #     self.stop_motion()
        # except Exception as e:
        #     rospy.logwarn(f"[XArmRobot] stop_motion best-effort failed: {e}")
        try:
            rospy.signal_shutdown("SIGINT")
        except Exception:
            pass

    # ---------------- callbacks / state ----------------
    def _robot_cb(self, msg: RobotMsg):
        self._latest_robot_msg = msg
        if self.debug:
            rospy.loginfo_throttle(
                0.1,
                f"[xarm] mode={int(msg.mode)} state={int(msg.state)} err={int(msg.err)} warn={int(msg.warn)} cmd={int(msg.cmdnum)}"
            )

    def wait_for_state(self, timeout_s: float = 5.0) -> RobotMsg:
        start = rospy.Time.now().to_sec()
        r = rospy.Rate(200)
        while not rospy.is_shutdown() and not self._stop_requested:
            if self._latest_robot_msg is not None:
                return self._latest_robot_msg
            if rospy.Time.now().to_sec() - start > timeout_s:
                raise RuntimeError(f"Timeout waiting for {ROBOT_TOPIC}")
            r.sleep()
        raise RuntimeError("Stopped (SIGINT) or ROS shutdown")

    # ---------------- internal call helper ----------------
    def _call(self, srv_name: str, fn, req=None) -> CallResult:
        if self._stop_requested:
            return CallResult(ok=False, ret=-1, message="stopped")
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

    # ---------------- state snapshots ----------------
    def _cur_joints(self) -> Optional[np.ndarray]:
        msg = self._latest_robot_msg
        if msg is None:
            return None
        try:
            return np.array(list(msg.angle), dtype=np.float32)
        except Exception:
            return None

    def _cur_pose6(self) -> Optional[np.ndarray]:
        msg = self._latest_robot_msg
        if msg is None:
            return None
        try:
            return np.array(list(msg.pose), dtype=np.float32)
        except Exception:
            return None

    def _ready_idle(self, msg: RobotMsg) -> bool:
        # Your controller often idles at state=2
        return int(msg.err) == 0 and int(msg.state) in (0, 2)

    # ---------------- goal-based blocking (ROS2-like) ----------------
    def _wait_reach_joints(self, goal: List[float], tol: float, timeout_s: float) -> bool:
        goal = np.array(goal, dtype=np.float32)
        start = rospy.Time.now().to_sec()
        r = rospy.Rate(200)

        while not rospy.is_shutdown() and not self._stop_requested:
            msg = self._latest_robot_msg
            if msg is None:
                r.sleep()
                continue

            if int(msg.err) != 0:
                return False

            cur = self._cur_joints()
            if cur is not None and cur.shape[0] >= goal.shape[0]:
                e = np.max(np.abs(cur[: goal.shape[0]] - goal))
                if e < tol:
                    return True

            if rospy.Time.now().to_sec() - start > timeout_s:
                return False

            r.sleep()
        return False

    def _wait_reach_pose(self, goal_pose6: List[float], tol_mm: float, tol_rad: float, timeout_s: float) -> bool:
        g = np.array(goal_pose6, dtype=np.float32)
        start = rospy.Time.now().to_sec()
        r = rospy.Rate(200)

        while not rospy.is_shutdown() and not self._stop_requested:
            msg = self._latest_robot_msg
            if msg is None:
                r.sleep()
                continue

            if int(msg.err) != 0:
                return False

            cur = self._cur_pose6()
            if cur is not None and cur.shape[0] >= 6:
                pos_err = np.max(np.abs(cur[:3] - g[:3]))
                ang_err = np.max(np.abs(_wrap_to_pi(cur[3:6] - g[3:6])))
                if pos_err < tol_mm and ang_err < tol_rad:
                    return True

            if rospy.Time.now().to_sec() - start > timeout_s:
                return False

            r.sleep()
        return False

    def _wait_idle(self, timeout_s: float = 5.0) -> bool:
        start = rospy.Time.now().to_sec()
        r = rospy.Rate(200)
        while not rospy.is_shutdown() and not self._stop_requested:
            msg = self._latest_robot_msg
            if msg is not None:
                if int(msg.err) != 0:
                    return False
                if self._ready_idle(msg):
                    return True
            if rospy.Time.now().to_sec() - start > timeout_s:
                return False
            r.sleep()
        return False

    # ---------------- 1) initialize ----------------
    def initialize(self) -> CallResult:
        r1 = self.set_mode(MODE_POSITION)
        if not r1.ok:
            return r1
        r2 = self.set_state(0)
        if not r2.ok:
            return r2
        if not self._wait_idle(timeout_s=5.0):
            msg = self._latest_robot_msg
            return CallResult(False, -1, f"not idle after init (state={getattr(msg,'state',None)} err={getattr(msg,'err',None)})")
        return self.home()

    # ---------------- 2) mode/state ----------------
    def set_mode(self, mode: int) -> CallResult:
        rospy.wait_for_service(self.services.set_mode)
        try:
            req = SetInt16Request()
            req.data = int(mode)
            res = self._set_mode(req)
            return _as_call_result(res)
        except rospy.ServiceException as e:
            return CallResult(False, -1, str(e))

    def set_state(self, state: int) -> CallResult:
        rospy.wait_for_service(self.services.set_state)
        try:
            req = SetInt16Request()
            req.data = int(state)
            res = self._set_state(req)
            return _as_call_result(res)
        except rospy.ServiceException as e:
            return CallResult(False, -1, str(e))

    def ensure_mode(self, mode: int, state: int = 0, timeout_s: float = 3.0) -> CallResult:
        self.wait_for_state(timeout_s=timeout_s)
        msg = self._latest_robot_msg

        if msg is not None and int(msg.mode) == int(mode) and int(msg.err) == 0 and int(msg.state) != 4:
            return CallResult(True, 0, "mode already ok")

        r1 = self.set_mode(mode)
        if not r1.ok:
            return r1
        r2 = self.set_state(state)
        if not r2.ok:
            return r2

        # wait until mode reflected and not error
        start = rospy.Time.now().to_sec()
        r = rospy.Rate(200)
        while not rospy.is_shutdown() and not self._stop_requested:
            m = self._latest_robot_msg
            if m is not None:
                if int(m.err) != 0:
                    return CallResult(False, -1, f"err after mode switch: {int(m.err)}")
                if int(m.mode) == int(mode) and int(m.state) != 4:
                    break
            if rospy.Time.now().to_sec() - start > timeout_s:
                last = self._latest_robot_msg
                return CallResult(False, -1, f"timeout waiting mode={mode} last(mode={getattr(last,'mode',None)} state={getattr(last,'state',None)})")
            r.sleep()

        # For position mode, wait for idle
        if int(mode) == int(MODE_POSITION):
            if not self._wait_idle(timeout_s=timeout_s):
                last = self._latest_robot_msg
                return CallResult(False, -1, f"not idle after mode0 (state={getattr(last,'state',None)} err={getattr(last,'err',None)})")

        return CallResult(True, 0, "mode set")

    # ---------------- optional stop motion ----------------
    # TODO: check if this is needed
    def stop_motion(self) -> CallResult:
        """
        Best-effort stop via /xarm/motion_ctrl (SetInt16) if available in config.
        Many setups interpret 0 as stop/pause.
        """
        if self._motion_ctrl is None or not hasattr(self.services, "motion_ctrl"):
            return CallResult(False, -1, "motion_ctrl not configured")
        try:
            rospy.wait_for_service(self.services.motion_ctrl, timeout=1.0)
            req = SetInt16Request()
            req.data = 0
            res = self._motion_ctrl(req)
            return _as_call_result(res)
        except Exception as e:
            return CallResult(False, -1, str(e))

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
            return CallResult(False, -1, str(e)), -1.0

    def move_gripper(self, pulse_pos: float) -> CallResult:
        try:
            pulse_pos = float(pulse_pos)
        except Exception:
            rospy.logwarn(f"[xArm] move_gripper: invalid pulse_pos={pulse_pos}, forcing to GRIPPER_MIN")
            pulse_pos = GRIPPER_MIN

        # cap instead of assert
        capped = min(max(pulse_pos, GRIPPER_MIN), GRIPPER_MAX)
        if capped != pulse_pos:
            rospy.logwarn(
                f"[xArm] move_gripper: pulse_pos {pulse_pos:.3f} out of range "
                f"[{GRIPPER_MIN}, {GRIPPER_MAX}], capped to {capped:.3f}"
            )
        pulse_pos = capped

        req = GripperMoveRequest()
        req.pulse_pos = pulse_pos
        return self._call(self.services.gripper_move, self._gripper_move, req)

    # ---------------- 4) home ----------------
    def home(self) -> CallResult:
        r = self.move_to_joint(HOME_JOINT)
        if not r.ok:
            return r
        return self.move_gripper(HOME_GRIPPER)

    # ---------------- 5) position motion (GOAL-BASED BLOCKING) ----------------
    def move_to_joint(
        self,
        joints: List[float],
        mvvelo: float = 0.35,
        mvacc: float = 7.0,
        mvtime: float = 0.0,
        timeout_s: float = 60.0,
        tol_rad: Optional[float] = None,
    ) -> CallResult:
        assert isinstance(joints, list) and len(joints) == 7
        tol = self.joint_tol_rad if tol_rad is None else float(tol_rad)

        rmode = self.ensure_mode(MODE_POSITION, state=0)
        if not rmode.ok:
            return rmode

        req = MoveRequest()
        req.pose = [float(x) for x in joints]
        req.mvvelo = float(mvvelo)
        req.mvacc = float(mvacc)
        req.mvtime = float(mvtime)
        if hasattr(req, "mvradii"):
            req.mvradii = 0.0

        out = self._call(self.services.move_joint, self._move_joint, req)
        if not out.ok:
            return out

        ok = self._wait_reach_joints(joints, tol=tol, timeout_s=float(timeout_s))
        if not ok:
            msg = self._latest_robot_msg
            return CallResult(False, -1, f"timeout waiting reach joints (state={getattr(msg,'state',None)} err={getattr(msg,'err',None)})")

        # optional extra settle: ensure idle
        self._wait_idle(timeout_s=2.0)
        return CallResult(True, 0, "done")

    def move_to_pose(
        self,
        pose6: List[float],
        mvvelo: float = 200.0,
        mvacc: float = 1000.0,
        mvtime: float = 0.0,
        timeout_s: float = 60.0,
        tol_mm: Optional[float] = None,
        tol_rad: Optional[float] = None,
    ) -> CallResult:
        assert isinstance(pose6, list) and len(pose6) == 6
        tol_mm_ = self.pose_tol_mm if tol_mm is None else float(tol_mm)
        tol_rad_ = self.pose_tol_rad if tol_rad is None else float(tol_rad)

        rmode = self.ensure_mode(MODE_POSITION, state=0)
        if not rmode.ok:
            return rmode

        req = MoveRequest()
        req.pose = [float(x) for x in pose6]
        req.mvvelo = float(mvvelo)
        req.mvacc = float(mvacc)
        req.mvtime = float(mvtime)
        if hasattr(req, "mvradii"):
            req.mvradii = 0.0

        out = self._call(self.services.move_line, self._move_line, req)
        if not out.ok:
            return out

        ok = self._wait_reach_pose(pose6, tol_mm=tol_mm_, tol_rad=tol_rad_, timeout_s=float(timeout_s))
        if not ok:
            msg = self._latest_robot_msg
            return CallResult(False, -1, f"timeout waiting reach pose (state={getattr(msg,'state',None)} err={getattr(msg,'err',None)})")

        self._wait_idle(timeout_s=2.0)
        return CallResult(True, 0, "done")

    # ---------------- 6) cartesian velocity (TIMED BLOCKING) ----------------
    def velo_move_line_timed(
        self,
        speeds6_mps: List[float],  # m/s + rad/s
        duration: float,
        is_tool_coord: bool = True,
        is_sync: bool = True,
        settle_idle: bool = False,
    ) -> CallResult:
        """
        /xarm/velo_move_line_timed (xarm_msgs/MoveVelocity)
        Completion is time-based: we sleep duration (+ small margin).
        Optionally wait idle afterwards if your controller returns to idle.
        """
        assert isinstance(speeds6_mps, list) and len(speeds6_mps) == 6

        rmode = self.ensure_mode(MODE_CART_VELO, state=0)
        if not rmode.ok:
            return rmode
        
        try:
            speeds6_xarm = speeds6_mps_to_xarm_units(
                speeds6_mps,
                max_lin_m_s=MAX_TCP_LIN_M_S,
                max_ang_rad_s=MAX_TCP_ANG_RAD_S,
                abs_sanity_lin_m_s=ABS_SANITY_LIN_M_S,
                abs_sanity_ang_rad_s=ABS_SANITY_ANG_RAD_S,
            )
        except ValueError as e:
            return CallResult(False, 1, f"bad speeds6: {e}")

        req = MoveVelocityRequest()
        req.speeds = [float(x) for x in speeds6_xarm]
        req.is_sync = bool(is_sync)
        req.is_tool_coord = bool(is_tool_coord)
        req.duration = float(duration)

        out = self._call(self.services.velo_move_line_timed, self._velo_move_line_timed, req)
        if not out.ok:
            return out

        # True "spin till done" for timed velocity is simply waiting out the duration.
        sleep_t = float(duration) + 0.02
        start = rospy.Time.now().to_sec()
        r = rospy.Rate(200)
        while not rospy.is_shutdown() and not self._stop_requested:
            if rospy.Time.now().to_sec() - start >= sleep_t:
                break
            r.sleep()

        if settle_idle:
            self._wait_idle(timeout_s=2.0)

        return CallResult(True, 0, "done")


    def move_servo_cart(self, pose6_mm_rpy: List[float], tool_coord: bool = False) -> CallResult:
        """
        /xarm/move_servo_cart:
        pose: [X(mm), Y(mm), Z(mm), R(rad), P(rad), Y(rad)]
        For base coordinate, mvvelo/mvacc/mvtime/mvradii are not effective => pass 0.
        For tool coordinate (firmware >=1.5.0), set mvtime=1 and pose is treated as tool-relative delta.
        """
        assert isinstance(pose6_mm_rpy, list) and len(pose6_mm_rpy) == 6

        # Servo_Cartesian runs in mode 1
        rmode = self.ensure_mode(MODE_SERVO_CART, state=0)
        if not rmode.ok:
            return rmode

        req = MoveRequest()
        req.pose = [float(x) for x in pose6_mm_rpy]
        # TODO: tune these values
        req.mvvelo = 100.0
        req.mvacc = 2000.0
        req.mvtime = 1.0 if tool_coord else 0.0
        if hasattr(req, "mvradii"):
            req.mvradii = 0.0

        return self._call(self.services.move_servo_cart, self._move_servo_cart, req)


    # ---------------- public state snapshot ----------------
    def get_state(self) -> XArmState:
        msg = self._latest_robot_msg

        ee_pose = None
        joints = None
        if msg is not None:
            try:
                ee_pose = list(msg.pose)
            except Exception:
                ee_pose = None
            try:
                joints = list(msg.angle)
            except Exception:
                joints = None

        _, gpos = self.get_gripper_state()
        gripper_qpos = None if gpos < 0 else gpos

        return XArmState(
            ee_pose=ee_pose,
            joint_angles=joints,
            gripper_qpos=gripper_qpos,
            mode=int(msg.mode) if msg is not None else None,
            state=int(msg.state) if msg is not None else None,
            err=int(msg.err) if msg is not None else None,
            warn=int(msg.warn) if msg is not None else None,
        )