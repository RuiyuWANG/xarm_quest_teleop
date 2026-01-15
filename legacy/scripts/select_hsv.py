import cv2
import numpy as np
# import pyrealsense2 as rs

# Initialize RealSense camera
# def setup_camera():
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#     pipeline.start(config)
#     sensor = pipeline.get_active_profile().get_device().first_color_sensor()
#     sensor.set_option(rs.option.enable_auto_white_balance, 0)
#     sensor.set_option(rs.option.white_balance, 3400)
#     return pipeline

# Global image buffer for click callback
current_image = cv2.imread("/home/ruiyuw/Codes/palm_main/palm/test.png")
current_image  =  cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
stop_requested = False  # ✅ Fix: add this line
# camera = setup_camera()

# Click callback
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and current_image is not None:
        bgr = current_image[y, x]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"[{x}, {y}] BGR: {bgr} → HSV: {hsv}")

# Set up OpenCV window and callback
cv2.namedWindow("Click to Get HSV")
cv2.setMouseCallback("Click to Get HSV", on_mouse_click)

try:
    while not stop_requested:
        # frames = camera.wait_for_frames()
        # color_frame = frames.get_color_frame()
        # if not color_frame:
        #     continue

        # color_image = np.asanyarray(color_frame.get_data())
        # current_image = color_image  # Share with mouse callback

        # Display live feed
        cv2.imshow("Click to Get HSV", current_image)
        if cv2.waitKey(1) == 27:  # ESC key
            stop_requested = True

finally:
    cv2.destroyAllWindows()
