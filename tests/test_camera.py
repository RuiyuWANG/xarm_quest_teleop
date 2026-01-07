# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
from img_utils import *

IMG_RAW_SIZE = (720, 1280, 3)  # (height, width, channels).
# Create a context object. This object owns the handles to all connected realsense devices
pipeline = setup_camera()

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        # color_image = color_image.reshape(IMG_RAW_SIZE)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)

        h, w = color_image.shape[:2]
        margin = int(w - h) // 2
        if margin >= 0:
            color_image = color_image[:, margin : margin + h]

        # Show image
        cv2.imshow('Agentview', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()