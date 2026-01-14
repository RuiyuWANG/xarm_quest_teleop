import os, signal, subprocess, threading, time
from typing import Optional, List
import rospy

class ManagedProcess:
    def __init__(self, name: str, cmd: List[str], workdir: Optional[str], pipe_output: bool):
        self.name = name
        self.cmd = cmd
        self.workdir = workdir
        self.pipe_output = pipe_output
        self.p: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.p = subprocess.Popen(
            self.cmd,
            cwd=self.workdir,
            stdout=subprocess.PIPE if self.pipe_output else None,
            stderr=subprocess.STDOUT if self.pipe_output else None,
            env=os.environ.copy(),
            preexec_fn=os.setsid,
            text=True,
            bufsize=1,
        )
        rospy.loginfo(f"[launch] started {self.name}: {' '.join(self.cmd)} (pid={self.p.pid})")
        if self.pipe_output and self.p.stdout is not None:
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()

    def _pump(self):
        assert self.p is not None and self.p.stdout is not None
        for line in self.p.stdout:
            line = line.rstrip("\n")
            if line:
                rospy.loginfo_throttle(0.2, f"[{self.name}] {line}")

    def alive(self) -> bool:
        return self.p is not None and self.p.poll() is None

    def stop(self):
        if self.p is None or self.p.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.p.pid), signal.SIGINT)
        except Exception as e:
            rospy.logwarn(f"[launch] stop {self.name} failed: {e}")
        t0 = time.time()
        while time.time() - t0 < 5.0 and self.alive():
            time.sleep(0.05)
        if self.alive():
            try:
                os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
            except Exception:
                pass

def wait_for_topic(topic: str, timeout_s: float) -> bool:
    t0 = rospy.Time.now().to_sec()
    r = rospy.Rate(20)
    while not rospy.is_shutdown():
        pubs = rospy.get_published_topics()
        if any(t[0] == topic for t in pubs):
            return True
        if rospy.Time.now().to_sec() - t0 > timeout_s:
            return False
        r.sleep()
    return False


def wait_for_service(name: str, timeout_s: float) -> bool:
    try:
        rospy.wait_for_service(name, timeout=timeout_s)
        return True
    except Exception:
        return False