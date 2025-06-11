# CloudGripper_Manipulation
1. Set up the robot
   Ufactory xArm, please refer to the official manual, https://www.ufactory.cc/wp-content/uploads/2023/05/xArm-User-Manual-V2.0.0.pdf
   1. Set up the robot, connect it to the adapter, and then to the electrical source. Connect the gripper.
   2. Connect your computer to the robot using NetworkWare or Wi-Fi. The robot's IP address is 192.168.1.244; set it to be your computer's IP address as well (see manual).
   3. Test your connection at UFactory Studio, IP:18333.
   4. Reset the initial position, TCP payload, TCP height offset (see manual). These are important for adjusting IK sover and safety constraints.
   5. Install xarm-python-sdk, pip install xarm-python-sdk, https://github.com/xArm-Developer/xArm-Python-SDK/tree/master.
   6. Test your robot and gripper (only supports xArm Gripper) by running test_robot.py
2. Set up the camera
   Real Sense, please refer to https://www.intelrealsense.com/get-started-depth-camera/
   1. Set up the camera and connect the camera to your computer.
   2. Install Intel® RealSense™ SDK 2.0, https://github.com/IntelRealSense/librealsense/blob/development/doc/distribution_linux.md
   3. Install Python wrapper and test your code, pip install pyrealsense2.
   4. Test your camera connection by running realsense-viewer or test_camera.py.
3. Camera Calibration
   
4. Evaluate your policy
