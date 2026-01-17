# src/io/process_manager.py
from __future__ import annotations

import atexit
from collections import deque
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Dict

import rospy


# ----------------------------- ManagedProcess -----------------------------

class ManagedProcess:
    """
    Starts a subprocess in its OWN process group (setsid), so we can terminate
    the whole tree (roslaunch + all node children) via killpg.

    Works well for:
      - roscore
      - roslaunch ...
      - rosrun ...
      - any long-running command
    """

    def __init__(
        self,
        name: str,
        cmd: List[str],
        workdir: Optional[str] = None,
        pipe_output: bool = True,
        env: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.cmd = list(cmd)
        self.workdir = workdir
        self.pipe_output = bool(pipe_output)
        self.env = env
        self.p: Optional[subprocess.Popen] = None

    def start(self):
        if self.p is not None and self.p.poll() is None:
            rospy.logwarn(f"[launch] {self.name} already running")
            return

        stdout = subprocess.PIPE if self.pipe_output else None
        stderr = subprocess.STDOUT if self.pipe_output else None

        # IMPORTANT: start in new process group so we can killpg later
        self.p = subprocess.Popen(
            self.cmd,
            cwd=self.workdir,
            stdout=stdout,
            stderr=stderr,
            env=self.env,
            text=True,
            preexec_fn=os.setsid,  # Linux: new session => new process group
        )
        rospy.loginfo(f"[launch] started {self.name}: {' '.join(self.cmd)} (pid={self.p.pid})")
        return self.p

    def is_running(self) -> bool:
        return self.p is not None and self.p.poll() is None

    def stop(self, timeout_s: float = 4.0):
        """
        Stop this process AND ITS CHILDREN.
        Strategy:
          1) SIGINT process group (like Ctrl+C) -> roslaunch shuts down nodes nicely
          2) wait timeout
          3) SIGTERM group
          4) wait short
          5) SIGKILL group
        """
        if self.p is None:
            return

        if self.p.poll() is not None:
            return

        try:
            pgid = os.getpgid(self.p.pid)
        except Exception:
            pgid = None

        def _kill_group(sig):
            if pgid is None:
                return
            try:
                os.killpg(pgid, sig)
            except Exception:
                pass

        # 1) try graceful
        _kill_group(signal.SIGINT)

        t0 = time.time()
        while time.time() - t0 < float(timeout_s):
            if self.p.poll() is not None:
                rospy.loginfo(f"[launch] stopped {self.name} (pid={self.p.pid})")
                return
            time.sleep(0.05)

        # 2) harder
        _kill_group(signal.SIGTERM)
        t1 = time.time()
        while time.time() - t1 < 1.0:
            if self.p.poll() is not None:
                rospy.loginfo(f"[launch] stopped {self.name} (pid={self.p.pid})")
                return
            time.sleep(0.05)

        # 3) last resort
        _kill_group(signal.SIGKILL)
        rospy.logwarn(f"[launch] force-killed {self.name} (pid={self.p.pid})")


# ----------------------------- ProcessSupervisor -----------------------------

class ProcessSupervisor:
    """
    Owns multiple ManagedProcess instances and guarantees cleanup when:
      - script exits normally
      - Ctrl+C (SIGINT)
      - killed (SIGTERM)

    Usage:
      sup = ProcessSupervisor()
      p = sup.start(ManagedProcess(...))
      ...
      rospy.on_shutdown(sup.stop_all)  # optional (still fine)
    """

    def __init__(self):
        self._procs: List[ManagedProcess] = []
        self._installed = False
        self._install_handlers()

    def start(self, p: ManagedProcess) -> ManagedProcess:
        p.start()
        self._procs.append(p)
        return p

    def register(self, p: ManagedProcess) -> ManagedProcess:
        # if caller started it manually
        self._procs.append(p)
        return p

    def stop_all(self):
        # Stop in reverse order of startup
        rospy.logwarn("[main] shutting down, stopping launched processes")
        for p in reversed(self._procs):
            try:
                p.stop()
            except Exception:
                pass

    def _install_handlers(self):
        if self._installed:
            return
        self._installed = True

        # Ensure stop_all runs on normal exit
        atexit.register(self.stop_all)

        # Also trap SIGINT/SIGTERM so even hard kill cleans up
        def _handler(signum, _frame):
            try:
                rospy.logwarn(f"[main] received signal {signum}, stopping processes...")
            except Exception:
                pass
            try:
                self.stop_all()
            finally:
                # re-raise default behavior
                raise SystemExit(0)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


# ----------------------------- Utilities (unchanged-ish) -----------------------------

def wait_for_topic(topic: str, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < float(timeout_s) and not rospy.is_shutdown():
        try:
            # fast check: topic list contains it
            topics = [t for (t, _) in rospy.get_published_topics()]
            if topic in topics:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def wait_for_service(service: str, timeout_s: float) -> bool:
    try:
        rospy.wait_for_service(service, timeout=float(timeout_s))
        return True
    except Exception:
        return False


# ----------------------------- FixedRateSelector -----------------------------
# If you still use it somewhere else, keep it here.
class FixedRateSelector:
    """
    Simple rate gate: should_take(t) returns True if enough time passed.
    Uses t (e.g., camera stamp) for consistent sampling.
    """
    def __init__(self, rate_hz: float):
        self.rate_hz = float(rate_hz)
        self.min_dt = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.0
        self._last_t: Optional[float] = None

    def should_take(self, t: float) -> bool:
        t = float(t)
        if self.min_dt <= 0:
            return True
        if self._last_t is None:
            self._last_t = t
            return True
        if (t - self._last_t) >= self.min_dt:
            self._last_t = t
            return True
        return False


class SampleRing:
    def __init__(self, keep_s: float = 1.0, maxlen: int = 2000):
        self.keep_s = float(keep_s)
        self.buf = deque(maxlen=maxlen)

    def push(self, sample):
        self.buf.append(sample)
        t_now = float(sample.stamp_sync)
        t_min = t_now - self.keep_s
        while self.buf and float(self.buf[0].stamp_sync) < t_min:
            self.buf.popleft()

    def nearest(self, t: float, window: float):
        if not self.buf:
            return None
        best = None
        best_dt = 1e9
        for s in self.buf:
            dt = abs(float(s.stamp_sync) - t)
            if dt < best_dt:
                best = s
                best_dt = dt
        if best is None or best_dt > float(window):
            return None
        return best