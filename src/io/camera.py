from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable
from collections import deque

import rospy
import message_filters
from sensor_msgs.msg import Image

@dataclass
class SyncedCameraMsgs:
    rgb: Image
    depth: Image


@dataclass
class CamPair:
    t: float
    rgb: Image
    depth: Image


class _PairBuf:
    def __init__(self, maxlen: int):
        self.buf = deque(maxlen=maxlen)

    def push(self, x: CamPair):
        self.buf.append(x)

    def drop_older_than(self, t_min: float):
        while self.buf and self.buf[0].t < t_min:
            self.buf.popleft()

    def nearest(self, t: float, window: float) -> Optional[CamPair]:
        if not self.buf:
            return None
        best = None
        best_dt = 1e9
        for x in self.buf:
            dt = abs(x.t - t)
            if dt < best_dt:
                best, best_dt = x, dt
        if best is None or best_dt > window:
            return None
        return best

class _RingImg:
    def __init__(self, maxlen: int):
        self.buf = deque(maxlen=maxlen)

    def push(self, msg: Image):
        self.buf.append(msg)

    def drop_older_than(self, t_min: float):
        while self.buf and self.buf[0].header.stamp.to_sec() < t_min:
            self.buf.popleft()

    def nearest(self, t: float, window: float) -> Optional[Image]:
        if not self.buf:
            return None
        best = None
        best_dt = 1e9
        for m in self.buf:
            dt = abs(m.header.stamp.to_sec() - t)
            if dt < best_dt:
                best, best_dt = m, dt
        if best is None or best_dt > window:
            return None
        return best

class CamRgbDepthSync:
    """
    Faster software-only sync for N cameras:

      Per camera:
        - Subscribe to rgb and depth directly (no ATS)
        - Build pairs by: on rgb -> nearest depth (within pair_slop_s)
        - Push CamPair(t=rgb_stamp, rgb, depth) into _pairs[name]

      Across cameras:
        - When ref cam pair arrives -> enqueue t_ref into pending queue
        - One worker thread tries to assemble complete set (nearest within tri_slop_s)
        - If incomplete: wait until max_wait_s then drop
        - No busy-wait spin loops

    Guarantees:
      - emits only COMPLETE sets (all cams have rgb+depth)
      - bounded latency (max_wait_s)
      - much lower CPU + fewer drops than ATS+busy-wait
    """

    def __init__(
        self,
        cameras: Dict[str, Dict[str, str]],   # name -> {"rgb_topic":..., "depth_topic":...}
        pair_slop_s: float,
        tri_slop_s: float,
        pair_buf_len: int,
        keep_s: float,
        max_wait_s: float,
        ref_cam: str,
        sub_queue_size: int = 10,   # IMPORTANT: not 1
    ):
        self.cameras = cameras
        self.pair_slop_s = float(pair_slop_s)
        self.tri_slop_s = float(tri_slop_s)
        self.pair_buf_len = int(pair_buf_len)
        self.keep_s = float(keep_s)
        self.max_wait_s = float(max_wait_s)
        self.ref_cam = ref_cam
        self.sub_queue_size = int(sub_queue_size)

        assert ref_cam in cameras, f"ref_cam={ref_cam} not in cameras keys"

        self._lock = threading.Lock()
        self._pairs: Dict[str, _PairBuf] = {k: _PairBuf(maxlen=self.pair_buf_len) for k in cameras.keys()}

        # per-cam raw buffers for pairing
        self._rgb_buf: Dict[str, _RingImg] = {k: _RingImg(maxlen=self.pair_buf_len) for k in cameras.keys()}
        self._dep_buf: Dict[str, _RingImg] = {k: _RingImg(maxlen=self.pair_buf_len) for k in cameras.keys()}

        # user callback: fn(t_ref: float, cams: Dict[str, SyncedCameraMsgs]) -> None
        self.on_set: Optional[Callable[[float, Dict[str, SyncedCameraMsgs]], None]] = None

        # pending ref timestamps
        self._pending = deque()
        self._pending_cv = threading.Condition()
        self._pending_deadline: Dict[float, float] = {}  # t_ref -> wall deadline

        self._subs = []
        for name, spec in cameras.items():
            rgb_t = spec["rgb_topic"]
            dep_t = spec["depth_topic"]

            self._subs.append(
                rospy.Subscriber(
                    rgb_t, Image,
                    callback=lambda m, n=name: self._on_rgb(m, n),
                    queue_size=self.sub_queue_size
                )
            )
            self._subs.append(
                rospy.Subscriber(
                    dep_t, Image,
                    callback=lambda m, n=name: self._on_depth(m, n),
                    queue_size=self.sub_queue_size
                )
            )
            rospy.loginfo(f"[CamRgbDepthSyncFast] {name} rgb={rgb_t} depth={dep_t}")

        self._worker = threading.Thread(target=self._pending_worker, daemon=True)
        self._worker.start()

        rospy.loginfo(
            f"[CamRgbDepthSyncFast] ref={self.ref_cam} pair_slop={self.pair_slop_s}s "
            f"tri_slop={self.tri_slop_s}s keep_s={self.keep_s}s max_wait={self.max_wait_s}s "
            f"pair_buf={self.pair_buf_len} sub_q={self.sub_queue_size}"
        )

    def _on_rgb(self, rgb: Image, name: str):
        t = rgb.header.stamp.to_sec()
        with self._lock:
            self._rgb_buf[name].push(rgb)
            # try to pair immediately using nearest depth
            dep = self._dep_buf[name].nearest(t, self.pair_slop_s)
            if dep is None:
                return
            # keep buffers bounded by time
            t_min = t - self.keep_s
            self._rgb_buf[name].drop_older_than(t_min)
            self._dep_buf[name].drop_older_than(t_min)

            # make pair
            self._pairs[name].push(CamPair(t=t, rgb=rgb, depth=dep))

        # if ref cam, schedule emit attempt
        if name == self.ref_cam:
            self._enqueue_ref(t_ref=t)

    def _on_depth(self, depth: Image, name: str):
        t = depth.header.stamp.to_sec()
        with self._lock:
            self._dep_buf[name].push(depth)
            # We do NOT pair on depth; pairing happens on rgb to ensure t=rgb_stamp
            t_min = t - self.keep_s
            self._dep_buf[name].drop_older_than(t_min)

    def _enqueue_ref(self, t_ref: float):
        now = time.time()
        dl = now + self.max_wait_s
        with self._pending_cv:
            self._pending.append(t_ref)
            self._pending_deadline[t_ref] = dl
            self._pending_cv.notify()

    def _pending_worker(self):
        while not rospy.is_shutdown():
            with self._pending_cv:
                while not self._pending and not rospy.is_shutdown():
                    self._pending_cv.wait(timeout=0.1)
                if rospy.is_shutdown():
                    return
                t_ref = self._pending.popleft()
                deadline = self._pending_deadline.pop(t_ref, time.time())

            # try assemble until deadline
            while time.time() < deadline and not rospy.is_shutdown():
                out = None
                with self._lock:
                    # drop old pairs across cams
                    t_min = t_ref - self.keep_s
                    for b in self._pairs.values():
                        b.drop_older_than(t_min)

                    ok = True
                    tmp: Dict[str, SyncedCameraMsgs] = {}
                    for cam_name, buf in self._pairs.items():
                        cp = buf.nearest(t_ref, self.tri_slop_s)
                        if cp is None:
                            ok = False
                            break
                        tmp[cam_name] = SyncedCameraMsgs(rgb=cp.rgb, depth=cp.depth)

                    if ok:
                        out = tmp

                if out is not None:
                    cb = self.on_set
                    if cb is not None:
                        cb(t_ref, out)
                    break

                # sleep a bit (no busy-wait)
                time.sleep(0.002)

            # if timed out: drop silently (bounded latency)
            continue
class TwoRgbSync:
    def __init__(
        self,
        cameras: Dict[str, Dict[str, str]],
        slop_s: float,
        queue_size: int = 60,
        sub_queue_size: int = 10,
    ):
        self.on_set = None
        self.names = list(cameras.keys())

        # KEEP references to subscribers on self
        self.subs = [
            message_filters.Subscriber(
                cameras[name]["rgb_topic"],
                Image,
                queue_size=sub_queue_size
            )
            for name in self.names
        ]

        # KEEP reference to synchronizer on self
        self.ats = message_filters.ApproximateTimeSynchronizer(
            self.subs,
            queue_size=queue_size,
            slop=slop_s,
            allow_headerless=False,
        )
        self.ats.registerCallback(self._cb)

        rospy.loginfo(f"[TwoRgbSync] subscribed to: {[cameras[n]['rgb_topic'] for n in self.names]}")

    def _cb(self, *imgs: Image):
        ts = [img.header.stamp.to_sec() for img in imgs]
        t_ref = sum(ts) / len(ts)
        out = {name: img for name, img in zip(self.names, imgs)}
        if self.on_set:
            self.on_set(t_ref, out)

# class TwoRgbSync:
#     def __init__(
#         self, 
#         cameras: Dict[str, Dict[str, str]], 
#         slop_s: float, 
#         queue_size: int = 60, 
#         sub_queue_size: int = 10
#     ):
#         self.on_set = None
#         self.names = list(cameras.keys())
#         subs = []
#         for name in self.names:
#             subs.append(message_filters.Subscriber(cameras[name]["rgb_topic"], Image, queue_size=sub_queue_size))
#         self.ats = message_filters.ApproximateTimeSynchronizer(
#             subs, queue_size=queue_size, slop=slop_s, allow_headerless=False
#         )
#         self.ats.registerCallback(self._cb)

#     def _cb(self, *imgs: Image):
#         ts = [img.header.stamp.to_sec() for img in imgs]
#         t_ref = sum(ts) / len(ts)
#         out = {name: img for name, img in zip(self.names, imgs)}
#         if self.on_set:
#             self.on_set(t_ref, out)
