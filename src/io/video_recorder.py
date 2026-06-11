import os
import threading
import queue
from typing import Optional, Tuple

import numpy as np

class AsyncVideoWriter:
    """
    Non-blocking video writer:
      - control loop calls enqueue(frame_rgb_uint8)
      - writer thread does encoding/IO
    Uses imageio/ffmpeg or Pillow based on the output extension.
    """
    def __init__(self, fps: float = 20.0):
        self.fps = float(fps)
        self._q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=500)
        self._thread: Optional[threading.Thread] = None
        self._writer = None
        self._running = False
        self._path: Optional[str] = None

    def start(self, path: str):
        self.stop()  # just in case
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._path = path
        self._running = True

        def _worker():
            if str(self._path).lower().endswith(".gif"):
                self._write_gif_worker()
                return

            try:
                import imageio.v2 as imageio
                self._writer = imageio.get_writer(self._path, fps=self.fps)
            except Exception:
                self._writer = None
                self._running = False
                return

            while True:
                item = self._q.get()
                if item is None:
                    break
                try:
                    # item: HWC RGB uint8
                    self._writer.append_data(item)
                except Exception:
                    pass

            try:
                self._writer.close()
            except Exception:
                pass

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def _write_gif_worker(self):
        try:
            from PIL import Image
        except Exception:
            self._running = False
            return

        frames = []
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                frames.append(Image.fromarray(item.astype(np.uint8, copy=False)))
            except Exception:
                pass

        if frames:
            duration_ms = int(round(1000.0 / max(1.0, self.fps)))
            try:
                frames[0].save(
                    self._path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration_ms,
                    loop=0,
                )
            except Exception:
                pass
        self._running = False

    def enqueue(self, frame_rgb_uint8: np.ndarray):
        if not self._running:
            return
        if frame_rgb_uint8 is None:
            return
        try:
            # avoid blocking control loop if queue is full
            self._q.put_nowait(frame_rgb_uint8)
        except queue.Full:
            pass

    def stop(self):
        if self._running:
            self._running = False
            try:
                self._q.put_nowait(None)
            except Exception:
                pass
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        self._thread = None
        self._writer = None
        self._path = None
