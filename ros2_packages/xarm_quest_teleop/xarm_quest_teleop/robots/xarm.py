from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from xarm_quest_teleop.ros import compat as rospy
from xarm_quest_teleop.configs.robot_config import (
    ABS_SANITY_ANG_RAD_S,
    ABS_SANITY_LIN_M_S,
    GRIPPER_MAX,
    GRIPPER_MIN,
    HOME_GRIPPER,
    HOME_JOINT,
    MAX_TCP_ANG_RAD_S,
    MAX_TCP_LIN_M_S,
    MODE_CART_VELO,
    MODE_POSITION,
    MODE_SERVO_CART,
    ROBOT_TOPIC,
    XArmServices,
)
from xarm_quest_teleop.utils.robot_utils import _wrap_to_pi, speeds6_mps_to_xarm_units

from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import (
    GetFloat32,
    GripperMove,
    MoveCartesian,
    MoveJoint,
    MoveVelocity,
    SetFloat32,
    SetInt16,
    SetInt16ById,
)


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
    """ROS2 xArm wrapper preserving the previous branch's high-level robot API."""

    def __init__(
        self,
        services: XArmServices = XArmServices(),
        wait_for_finish_param: bool = True,
        auto_init: bool = True,
        debug: bool = True,
        joint_tol_rad: float = 0.01,
        pose_tol_mm: float = 1.0,
        pose_tol_rad: float = 0.03,
    ):
        del wait_for_finish_param
        self.services = services
        self.debug = bool(debug)
        self.joint_tol_rad = float(joint_tol_rad)
        self.pose_tol_mm = float(pose_tol_mm)
        self.pose_tol_rad = float(pose_tol_rad)
        self._latest_robot_msg: Optional[RobotMsg] = None
        self._stop_requested = False

        signal.signal(signal.SIGINT, self._on_sigint)

        rospy.Subscriber(ROBOT_TOPIC, RobotMsg, self._robot_cb, queue_size=10)

        self._set_mode = rospy.ServiceProxy(self.services.set_mode, SetInt16)
        self._set_state = rospy.ServiceProxy(self.services.set_state, SetInt16)
        self._motion_enable = rospy.ServiceProxy(self.services.motion_enable, SetInt16ById)
        self._move_joint = rospy.ServiceProxy(self.services.move_joint, MoveJoint)
        self._move_line = rospy.ServiceProxy(self.services.move_line, MoveCartesian)
        self._move_servo_cartesian = rospy.ServiceProxy(self.services.move_servo_cart, MoveCartesian)
        self._velo_move_line_timed = rospy.ServiceProxy(self.services.velo_move_line_timed, MoveVelocity)
        self._gripper_move = rospy.ServiceProxy(self.services.gripper_move, GripperMove)
        self._gripper_state = rospy.ServiceProxy(self.services.gripper_state, GetFloat32)
        self._set_gripper_mode = rospy.ServiceProxy(self.services.set_gripper_mode, SetInt16)
        self._set_gripper_enable = rospy.ServiceProxy(self.services.set_gripper_enable, SetInt16)
        self._set_gripper_speed = rospy.ServiceProxy(self.services.set_gripper_speed, SetFloat32)

        self.wait_for_state(timeout_s=10.0)
        if auto_init:
            out = self.initialize()
            if not out.ok:
                raise RuntimeError(f"initialize failed: ret={out.ret} msg={out.message}")
        rospy.loginfo("[XArmRobot] initialized for ROS2 xarm_api")

    def _on_sigint(self, _signum, _frame):
        if self._stop_requested:
            return
        self._stop_requested = True
        rospy.logwarn("[XArmRobot] SIGINT received; requesting ROS shutdown")
        rospy.signal_shutdown("SIGINT")

    def _robot_cb(self, msg: RobotMsg):
        self._latest_robot_msg = msg
        if self.debug:
            rospy.loginfo_throttle(
                0.5,
                f"[xarm] mode={int(msg.mode)} state={int(msg.state)} "
                f"err={int(msg.err)} warn={int(msg.warn)} cmd={int(msg.cmdnum)}",
            )

    def _call(self, srv_name: str, fn, req=None) -> CallResult:
        if self._stop_requested:
            return CallResult(False, -1, "stopped")
        try:
            res = fn(req) if req is not None else fn()
            out = _as_call_result(res)
            if not out.ok:
                rospy.logwarn(f"[{srv_name}] failed ret={out.ret} msg={out.message}")
            return out
        except Exception as exc:
            rospy.logerr(f"[{srv_name}] service call failed: {exc}")
            return CallResult(False, -1, str(exc))

    def wait_for_state(self, timeout_s: float = 5.0) -> RobotMsg:
        start = rospy.Time.now().to_sec()
        rate = rospy.Rate(200)
        while not rospy.is_shutdown() and not self._stop_requested:
            if self._latest_robot_msg is not None:
                return self._latest_robot_msg
            if rospy.Time.now().to_sec() - start > float(timeout_s):
                raise RuntimeError(f"Timeout waiting for {ROBOT_TOPIC}")
            rate.sleep()
        raise RuntimeError("Stopped or ROS shutdown")

    def _cur_joints(self) -> Optional[np.ndarray]:
        msg = self._latest_robot_msg
        if msg is None:
            return None
        return np.asarray(list(msg.angle), dtype=np.float32)

    def _cur_pose6(self) -> Optional[np.ndarray]:
        msg = self._latest_robot_msg
        if msg is None:
            return None
        return np.asarray(list(msg.pose), dtype=np.float32)

    @staticmethod
    def _ready_idle(msg: RobotMsg) -> bool:
        return int(msg.err) == 0 and int(msg.state) in (0, 2)

    def _wait_idle(self, timeout_s: float = 5.0) -> bool:
        start = rospy.Time.now().to_sec()
        rate = rospy.Rate(200)
        while not rospy.is_shutdown() and not self._stop_requested:
            msg = self._latest_robot_msg
            if msg is not None:
                if int(msg.err) != 0:
                    return False
                if self._ready_idle(msg):
                    return True
            if rospy.Time.now().to_sec() - start > float(timeout_s):
                return False
            rate.sleep()
        return False

    def _wait_reach_joints(self, goal: List[float], tol: float, timeout_s: float) -> bool:
        goal_arr = np.asarray(goal, dtype=np.float32)
        start = rospy.Time.now().to_sec()
        rate = rospy.Rate(200)
        while not rospy.is_shutdown() and not self._stop_requested:
            msg = self._latest_robot_msg
            if msg is not None and int(msg.err) != 0:
                return False
            cur = self._cur_joints()
            if cur is not None and cur.shape[0] >= goal_arr.shape[0]:
                if float(np.max(np.abs(cur[: goal_arr.shape[0]] - goal_arr))) < float(tol):
                    return True
            if rospy.Time.now().to_sec() - start > float(timeout_s):
                return False
            rate.sleep()
        return False

    def _wait_reach_pose(self, goal_pose6: List[float], tol_mm: float, tol_rad: float, timeout_s: float) -> bool:
        goal = np.asarray(goal_pose6, dtype=np.float32)
        start = rospy.Time.now().to_sec()
        rate = rospy.Rate(200)
        while not rospy.is_shutdown() and not self._stop_requested:
            msg = self._latest_robot_msg
            if msg is not None and int(msg.err) != 0:
                return False
            cur = self._cur_pose6()
            if cur is not None and cur.shape[0] >= 6:
                pos_err = float(np.max(np.abs(cur[:3] - goal[:3])))
                ang_err = float(np.max(np.abs(_wrap_to_pi(cur[3:6] - goal[3:6]))))
                if pos_err < float(tol_mm) and ang_err < float(tol_rad):
                    return True
            if rospy.Time.now().to_sec() - start > float(timeout_s):
                return False
            rate.sleep()
        return False

    def motion_enable(self, enable: bool = True, servo_id: int = 8) -> CallResult:
        req = SetInt16ById.Request()
        req.id = int(servo_id)
        req.data = 1 if enable else 0
        return self._call(self.services.motion_enable, self._motion_enable, req)

    def set_mode(self, mode: int) -> CallResult:
        req = SetInt16.Request()
        req.data = int(mode)
        return self._call(self.services.set_mode, self._set_mode, req)

    def set_state(self, state: int) -> CallResult:
        req = SetInt16.Request()
        req.data = int(state)
        return self._call(self.services.set_state, self._set_state, req)

    def ensure_mode(self, mode: int, state: int = 0, timeout_s: float = 3.0) -> CallResult:
        self.wait_for_state(timeout_s=timeout_s)
        msg = self._latest_robot_msg
        if msg is not None and int(msg.mode) == int(mode) and int(msg.err) == 0 and int(msg.state) != 4:
            return CallResult(True, 0, "mode already ok")
        out = self.set_mode(mode)
        if not out.ok:
            return out
        out = self.set_state(state)
        if not out.ok:
            return out
        start = rospy.Time.now().to_sec()
        rate = rospy.Rate(200)
        while not rospy.is_shutdown() and not self._stop_requested:
            msg = self._latest_robot_msg
            if msg is not None:
                if int(msg.err) != 0:
                    return CallResult(False, -1, f"err after mode switch: {int(msg.err)}")
                if int(msg.mode) == int(mode) and int(msg.state) != 4:
                    return CallResult(True, 0, "mode set")
            if rospy.Time.now().to_sec() - start > float(timeout_s):
                return CallResult(False, -1, f"timeout waiting for mode={mode}")
            rate.sleep()
        return CallResult(False, -1, "stopped")

    def setup_gripper(self, enable: bool = True, mode: int = 0, speed: float = 1500.0) -> CallResult:
        req_enable = SetInt16.Request()
        req_enable.data = 1 if enable else 0
        out = self._call(self.services.set_gripper_enable, self._set_gripper_enable, req_enable)
        if not out.ok:
            return out
        req_mode = SetInt16.Request()
        req_mode.data = int(mode)
        out = self._call(self.services.set_gripper_mode, self._set_gripper_mode, req_mode)
        if not out.ok:
            return out
        req_speed = SetFloat32.Request()
        req_speed.data = float(speed)
        return self._call(self.services.set_gripper_speed, self._set_gripper_speed, req_speed)

    def initialize(self) -> CallResult:
        out = self.motion_enable(True)
        if not out.ok:
            return out
        out = self.set_mode(MODE_POSITION)
        if not out.ok:
            return out
        out = self.set_state(0)
        if not out.ok:
            return out
        self.setup_gripper(enable=True, mode=0, speed=1500.0)
        if not self._wait_idle(timeout_s=5.0):
            return CallResult(False, -1, "not idle after init")
        return self.home()

    def get_gripper_state(self) -> Tuple[CallResult, float]:
        try:
            res = self._gripper_state(GetFloat32.Request())
            out = _as_call_result(res)
            return out, float(getattr(res, "data", -1.0)) if out.ok else -1.0
        except Exception as exc:
            return CallResult(False, -1, str(exc)), -1.0

    def move_gripper(self, pulse_pos: float) -> CallResult:
        try:
            pulse_pos = float(pulse_pos)
        except Exception:
            rospy.logwarn(f"[xArm] invalid gripper target {pulse_pos}; using {GRIPPER_MIN}")
            pulse_pos = GRIPPER_MIN
        pulse_pos = float(min(max(pulse_pos, GRIPPER_MIN), GRIPPER_MAX))
        req = GripperMove.Request()
        req.pos = pulse_pos
        req.wait = False
        req.timeout = 10.0
        return self._call(self.services.gripper_move, self._gripper_move, req)

    def home(self) -> CallResult:
        out = self.move_to_joint(HOME_JOINT)
        if not out.ok:
            return out
        return self.move_gripper(HOME_GRIPPER)

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
        out = self.ensure_mode(MODE_POSITION, state=0)
        if not out.ok:
            return out
        req = MoveJoint.Request()
        req.angles = [float(x) for x in joints]
        req.speed = float(mvvelo)
        req.acc = float(mvacc)
        req.mvtime = float(mvtime)
        req.wait = False
        req.timeout = float(timeout_s)
        req.radius = -1.0
        out = self._call(self.services.move_joint, self._move_joint, req)
        if not out.ok:
            return out
        if not self._wait_reach_joints(joints, tol_rad or self.joint_tol_rad, timeout_s):
            return CallResult(False, -1, "timeout waiting to reach joint goal")
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
        out = self.ensure_mode(MODE_POSITION, state=0)
        if not out.ok:
            return out
        req = MoveCartesian.Request()
        req.pose = [float(x) for x in pose6]
        req.speed = float(mvvelo)
        req.acc = float(mvacc)
        req.mvtime = float(mvtime)
        req.wait = False
        req.timeout = float(timeout_s)
        req.radius = -1.0
        req.is_tool_coord = False
        req.relative = False
        req.motion_type = 0
        out = self._call(self.services.move_line, self._move_line, req)
        if not out.ok:
            return out
        if not self._wait_reach_pose(pose6, tol_mm or self.pose_tol_mm, tol_rad or self.pose_tol_rad, timeout_s):
            return CallResult(False, -1, "timeout waiting to reach pose goal")
        self._wait_idle(timeout_s=2.0)
        return CallResult(True, 0, "done")

    def velo_move_line_timed(
        self,
        speeds6_mps: List[float],
        duration: float,
        is_tool_coord: bool = True,
        is_sync: bool = True,
        settle_idle: bool = False,
    ) -> CallResult:
        assert isinstance(speeds6_mps, list) and len(speeds6_mps) == 6
        out = self.ensure_mode(MODE_CART_VELO, state=0)
        if not out.ok:
            return out
        try:
            speeds6_xarm = speeds6_mps_to_xarm_units(
                speeds6_mps,
                max_lin_m_s=MAX_TCP_LIN_M_S,
                max_ang_rad_s=MAX_TCP_ANG_RAD_S,
                abs_sanity_lin_m_s=ABS_SANITY_LIN_M_S,
                abs_sanity_ang_rad_s=ABS_SANITY_ANG_RAD_S,
            )
        except ValueError as exc:
            return CallResult(False, 1, f"bad speeds6: {exc}")
        req = MoveVelocity.Request()
        req.speeds = [float(x) for x in speeds6_xarm]
        req.is_sync = bool(is_sync)
        req.is_tool_coord = bool(is_tool_coord)
        req.duration = float(duration)
        out = self._call(self.services.velo_move_line_timed, self._velo_move_line_timed, req)
        if not out.ok:
            return out
        time.sleep(float(duration) + 0.02)
        if settle_idle:
            self._wait_idle(timeout_s=2.0)
        return CallResult(True, 0, "done")

    def move_servo_cart(self, pose6_mm_rpy: List[float], tool_coord: bool = False) -> CallResult:
        assert isinstance(pose6_mm_rpy, list) and len(pose6_mm_rpy) == 6
        out = self.ensure_mode(MODE_SERVO_CART, state=0)
        if not out.ok:
            return out
        req = MoveCartesian.Request()
        req.pose = [float(x) for x in pose6_mm_rpy]
        req.speed = 100.0
        req.acc = 2000.0
        req.mvtime = 1.0 if tool_coord else 0.0
        req.wait = False
        req.timeout = -1.0
        req.radius = -1.0
        req.is_tool_coord = bool(tool_coord)
        req.relative = False
        req.motion_type = 0
        return self._call(self.services.move_servo_cart, self._move_servo_cartesian, req)

    def stop_motion(self) -> CallResult:
        return self.motion_enable(False)

    def get_state(self) -> XArmState:
        msg = self._latest_robot_msg
        ee_pose = None
        joints = None
        mode = state = err = warn = None
        if msg is not None:
            try:
                ee_pose = list(msg.pose)
                joints = list(msg.angle)
                mode = int(msg.mode)
                state = int(msg.state)
                err = int(msg.err)
                warn = int(msg.warn)
            except Exception:
                pass
        _, gripper = self.get_gripper_state()
        return XArmState(
            ee_pose=ee_pose,
            joint_angles=joints,
            gripper_qpos=gripper if gripper >= 0.0 else None,
            mode=mode,
            state=state,
            err=err,
            warn=warn,
        )


XArmRos2Robot = XArmRobot
