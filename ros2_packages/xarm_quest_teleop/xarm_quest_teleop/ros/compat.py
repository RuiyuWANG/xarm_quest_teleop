from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
except Exception:  # pragma: no cover - lets pure unit tests run without ROS2.
    rclpy = None
    MultiThreadedExecutor = None
    Node = Any
    Parameter = None


_node: Optional[Node] = None
_executor: Optional[MultiThreadedExecutor] = None
_shutdown = False
_shutdown_hooks: List[Callable[[], None]] = []
_throttle_last: Dict[Tuple[str, str], float] = {}
_argv_params: Dict[str, Any] = {}


def _parse_value(raw: str) -> Any:
    text = str(raw)
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _scan_argv_params(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    argv = list(sys.argv if argv is None else argv)
    params: Dict[str, Any] = {}
    i = 0
    while i < len(argv):
        item = argv[i]
        if item.startswith("_") and ":=" in item:
            key, val = item.split(":=", 1)
            params[key.lstrip("_")] = _parse_value(val)
        elif item == "-p" and i + 1 < len(argv) and ":=" in argv[i + 1]:
            key, val = argv[i + 1].split(":=", 1)
            params[key.lstrip("_").lstrip("/")] = _parse_value(val)
            i += 1
        elif item.startswith("--param") and ":=" in item:
            key, val = item.split("=", 1)[-1].split(":=", 1)
            params[key.lstrip("_").lstrip("/")] = _parse_value(val)
        i += 1
    return params


def _param_name(name: str) -> str:
    out = str(name)
    if out.startswith("~"):
        out = out[1:]
    return out.lstrip("/")


def _require_node() -> Node:
    if _node is None:
        init_node("xarm_quest_teleop_node")
    assert _node is not None
    return _node


def get_node() -> Node:
    return _require_node()


def init_node(name: str, anonymous: bool = False, argv: Optional[List[str]] = None, **_: Any) -> Node:
    del anonymous
    global _node, _executor, _shutdown, _argv_params
    if rclpy is None:
        _shutdown = False
        _argv_params = _scan_argv_params(argv)
        return None  # type: ignore[return-value]
    if not rclpy.ok():
        rclpy.init(args=argv)
    if _node is None:
        _node = rclpy.create_node(name)
    if _executor is None:
        _executor = MultiThreadedExecutor()
        _executor.add_node(_node)
    _argv_params = _scan_argv_params(argv)
    _shutdown = False
    return _node


def spin() -> None:
    if rclpy is None:
        while not _shutdown:
            time.sleep(0.1)
        return
    if _executor is None:
        _require_node()
    assert _executor is not None
    try:
        _executor.spin()
    finally:
        shutdown()


def spin_once(timeout_sec: float = 0.1) -> None:
    if rclpy is None:
        time.sleep(timeout_sec)
        return
    if _executor is None:
        _require_node()
    assert _executor is not None
    _executor.spin_once(timeout_sec=timeout_sec)


def shutdown() -> None:
    global _shutdown, _node, _executor
    if _shutdown:
        return
    _shutdown = True
    for hook in reversed(_shutdown_hooks):
        try:
            hook()
        except Exception:
            pass
    if rclpy is not None and rclpy.ok():
        try:
            if _executor is not None and _node is not None:
                _executor.remove_node(_node)
        except Exception:
            pass
        try:
            if _node is not None:
                _node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
    _node = None
    _executor = None


def signal_shutdown(reason: str = "") -> None:
    logwarn(reason or "shutdown requested")
    shutdown()


def is_shutdown() -> bool:
    if rclpy is None:
        return _shutdown
    return _shutdown or not rclpy.ok()


def ok() -> bool:
    return not is_shutdown()


def sleep(duration: float) -> None:
    time.sleep(float(duration))


def on_shutdown(fn: Callable[[], None]) -> None:
    _shutdown_hooks.append(fn)


class _Stamp:
    def __init__(self, sec: float):
        self._sec = float(sec)

    def to_sec(self) -> float:
        return self._sec

    def to_msg(self):
        if rclpy is None:
            return None
        return _require_node().get_clock().now().to_msg()


class Time:
    @staticmethod
    def now() -> _Stamp:
        if rclpy is None or _node is None:
            return _Stamp(time.time())
        return _Stamp(float(_node.get_clock().now().nanoseconds) / 1e9)


class Duration:
    def __init__(self, secs: float):
        self.secs = float(secs)

    def to_sec(self) -> float:
        return self.secs


class Rate:
    def __init__(self, hz: float):
        self.period = 1.0 / max(float(hz), 1e-9)

    def sleep(self) -> None:
        time.sleep(self.period)


class Timer:
    def __init__(self, duration: Duration, callback: Callable[[Any], None], oneshot: bool = False):
        node = _require_node()
        self._oneshot = bool(oneshot)

        def _cb():
            callback(None)
            if self._oneshot:
                self.shutdown()

        self._timer = node.create_timer(float(duration.to_sec()), _cb)

    def shutdown(self) -> None:
        try:
            self._timer.cancel()
            self._timer.destroy()
        except Exception:
            pass


class Publisher:
    def __init__(self, topic: str, msg_type: Any, queue_size: int = 10):
        self._pub = _require_node().create_publisher(msg_type, topic, int(queue_size))

    def publish(self, msg: Any) -> None:
        self._pub.publish(msg)


class Subscriber:
    def __init__(self, topic: str, msg_type: Any, callback: Callable[[Any], None], queue_size: int = 10):
        self._sub = _require_node().create_subscription(msg_type, topic, callback, int(queue_size))


class ServiceException(Exception):
    pass


class ServiceProxy:
    def __init__(self, service_name: str, srv_type: Any):
        self.service_name = service_name
        self.srv_type = srv_type
        self._client = _require_node().create_client(srv_type, service_name)

    def __call__(self, req: Optional[Any] = None) -> Any:
        if req is None:
            req = self.srv_type.Request()
        if not self._client.wait_for_service(timeout_sec=5.0):
            raise ServiceException(f"service unavailable: {self.service_name}")
        future = self._client.call_async(req)
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        while not event.wait(timeout=0.05):
            if is_shutdown():
                raise ServiceException(f"shutdown while waiting for {self.service_name}")
            try:
                if _executor is not None:
                    _executor.spin_once(timeout_sec=0.01)
            except Exception:
                pass
        exc = future.exception()
        if exc is not None:
            raise ServiceException(str(exc))
        return future.result()


def wait_for_service(service: str, timeout: Optional[float] = None) -> None:
    node = _require_node()
    deadline = None if timeout is None else time.time() + float(timeout)
    while ok():
        names = {name for name, _types in node.get_service_names_and_types()}
        if service in names:
            return
        if deadline is not None and time.time() >= deadline:
            raise ServiceException(f"timeout waiting for service {service}")
        time.sleep(0.1)


def get_published_topics() -> List[Tuple[str, str]]:
    node = _require_node()
    return [(topic, ",".join(types)) for topic, types in node.get_topic_names_and_types()]


def _declare_if_needed(name: str, default: Any) -> None:
    if rclpy is None:
        return
    node = _require_node()
    if not node.has_parameter(name):
        try:
            node.declare_parameter(name, default)
        except Exception:
            pass


def has_param(name: str) -> bool:
    pname = _param_name(name)
    if pname in _argv_params:
        return True
    if rclpy is None or _node is None:
        return False
    return _node.has_parameter(pname)


def get_param(name: str, default: Any = None) -> Any:
    pname = _param_name(name)
    if pname in _argv_params:
        return _argv_params[pname]
    _declare_if_needed(pname, default)
    if rclpy is None or _node is None:
        return default
    try:
        value = _node.get_parameter(pname).value
        return default if value is None else value
    except Exception:
        return default


def set_param(name: str, value: Any) -> None:
    pname = _param_name(name)
    if rclpy is None:
        _argv_params[pname] = value
        return
    node = _require_node()
    if not node.has_parameter(pname):
        node.declare_parameter(pname, value)
    else:
        node.set_parameters([Parameter(pname, value=value)])


def _log(level: str, msg: str) -> None:
    if rclpy is None or _node is None:
        print(f"[{level}] {msg}", flush=True)
        return
    logger = _node.get_logger()
    getattr(logger, level.lower())(str(msg))


def _log_throttle(level: str, period: float, msg: str) -> None:
    key = (level, str(msg))
    now = time.time()
    last = _throttle_last.get(key, 0.0)
    if now - last >= float(period):
        _throttle_last[key] = now
        _log(level, msg)


def loginfo(msg: str) -> None:
    _log("info", msg)


def logwarn(msg: str) -> None:
    _log("warn", msg)


def logerr(msg: str) -> None:
    _log("error", msg)


def logdebug(msg: str) -> None:
    _log("debug", msg)


def loginfo_throttle(period: float, msg: str) -> None:
    _log_throttle("info", period, msg)


def logwarn_throttle(period: float, msg: str) -> None:
    _log_throttle("warn", period, msg)


def logerr_throttle(period: float, msg: str) -> None:
    _log_throttle("error", period, msg)


def namespace() -> str:
    if rclpy is None or _node is None:
        return "/"
    return _node.get_namespace()


def get_namespace() -> str:
    return namespace()
