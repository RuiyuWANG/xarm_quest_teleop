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
   Generate checkerboard: https://calib.io/pages/camera-calibration-pattern-generator?srsltid=AfmBOor0sQvMQBZJ_ymCCees492RmNStF6MFV4uo_5e7SV_rAkcbacIO
4. Evaluate your policy

ROS2 Setup
1. install ros2, e.g. humble, https://docs.ros.org/en/foxy/Releases/Release-Humble-Hawksbill.html
2. Set the package for the robot and camera: https://github.com/xArm-Developer/xarm_ros2, https://github.com/IntelRealSense/realsense-ros
3. source ~/ros2_ws/install/setup.zsh
4. Enable service by modifying /share/xarm_api/config/xarm_params.yaml in ros2 ws
5. Launch robot: roslaunch xarm_bringup xarm7_server.launch robot_ip:=192.168.1.241 report_type:=dev
6. launch camera: ros2 launch realsense2_camera rs_launch.py depth_module.enable:=false
7. run your code

Aruco marker
1. https://chev.me/arucogen/
