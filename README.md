# VR Teleop Hardware
1. Set up the robot

   Ufactory xArm, please refer to the official manual, https://www.ufactory.cc/wp-content/uploads/2023/05/xArm-User-Manual-V2.0.0.pdf
   1. Set up the robot, connect it to the adapter, and then to the electrical source. Connect the gripper.
   2. Connect your computer and robot to the same router. Set PC IPV4 ip to be in 192.168.1.0 - 192.168.1.255.
   Get PC access to internet with Wifi, perferrably seted by another router.
   3. Test your connection at UFactory Studio, IP:18333.
   4. Reset the initial position, TCP payload, TCP height offset (see manual). These are important for adjusting IK sover and safety constraints.

2. Set up Meta Quest 2

   1. Download Meta Horizon app on your mobile phone. Login with email. Pay attention to enter true age when register, otherwise some repos are not accessible to people under 18.
   2. Reboot Quest to sign in as owner. Press open and volume - at the same time. Connect Quest to the same WiFi of PC, and connect to app.
   3. In app, enable developer mode.
   4. Check Quest network IP and ping IP on PC to check connectivity.
   5. download the teleoperation app following, https://quest2ros.github.io/

3. Setup cameras.

   Connect the cameras to the PC, perferrably one to the motherboard and one to the front panel. Connect with USE 3.0 wire & port if it is necessary.

# ROS Set-up
1. Docker
   1. Start a persistent ROS1 docker container with full /dev access for USB and RealSense connection, X11 enabled for RViz, --gpus all to allow docker access to GPU
   ```
   docker run -it \
   --name ros1_noetic \
   --network host \
   --privileged \
   --gpus all \
   -v /dev:/dev \
   -v /run/udev:/run/udev:ro \
   -v ~/docker_shared/catkin_ws:/root/catkin_ws \
   -e DISPLAY=$DISPLAY \
   -e QT_X11_NO_MITSHM=1 \
   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
   ros:noetic \
   bash -c "apt update && apt install -y zsh && chsh -s /usr/bin/zsh && zsh"
   ```

   2. Start container
   Allow the docker access to display (IMPORTANT for visualization like rviz and rqt)
   ```
   xhost +local:docker

   ```
   ```
   sudo docker start -ai ros1_noetic
   ```

   open new shell in the running docker
   ```
   sudo docker exec -it ros1_noetic zsh
   ```
   
   source:
   ```
   source /opt/ros/noetic/setup.zsh 2>/dev/null || source /opt/ros/noetic/setup.bash
   source ~/catkin_ws/devel/setup.zsh 2>/dev/null || source ~/catkin_ws/devel/setup.bash
   ```

2. Quest: https://quest2ros.github.io/
   Follow: 
   ```
   roslaunch ros_tcp_endpoint endpoint.launch tcp_ip:=192.168.0.182 tcp_port:=10000
   ```

   ```
   rosrun quest2ros ros2quest.py
   ```

   check connection
   ```
   rostopic list
   ```

3. Robot
   Follow: https://github.com/xarm-developer/xarm_ros

   Test robot connection:
   ```
   roslaunch xarm_bringup xarm7_server.launch robot_ip:=192.168.1.241 report_type:=dev add_gripper:=true
   ```

   check Topic Register:
   ```
   rostopic list
   ```

   check connection
   ```
   rostopic echo /xarm/xarm_states
   ```

   check robot movement
   ```
   rosservice call /xarm/move_joint "{pose: [0,0,0,0,0,0,0], mvvelo: 0.35, mvacc: 7, mvtime: 0, wait: 0}"
   ```

   check gripper
   ```
   rosservice call /xarm/gripper_move 500

   ```

4. Camera (RealSense)
   Will need to build from source, to get a 2.55 + librealsense for D405 camera
   ```
   cd /root
   git clone https://github.com/IntelRealSense/librealsense.git
   cd librealsense
   git checkout v2.56.3
   mkdir -p build && cd build
   cmake .. -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false
   make -j"$(nproc)"
   make install
   ldconfig
   ```
   And ROS1 version of realsense-ros
   ```
   cd ~/catkin_ws/src
   git clone https://github.com/IntelRealSense/realsense-ros.git
   cd realsense-ros
   git checkout noetic-development
   ```
   Launch camera with name to differentiate cameras, replace it with your serial numbers

   ```
   # D405
   roslaunch realsense2_camera rs_camera.launch \
   serial_no:=230322271104 camera:=d405 color_width:=848 color_height:=480 color_fps:=30 align_depth:=True enable_sync:=True 

   # D435i front
   roslaunch realsense2_camera rs_camera.launch \
   serial_no:=335522071488 camera:=d435i_front color_width:=848 color_height:=480 color_fps:=30 align_depth:=True enable_sync:=True 

   # D435i shoulder
   roslaunch realsense2_camera rs_camera.launch \
   serial_no:=233522073481 camera:=d435i_shoulder color_width:=848 color_height:=480 color_fps:=30 align_depth:=True enable_sync:=True 
   ```

   Check serial numbers
   ```
   rs-enumerate-devices
   ```

   [Optional] Get RViz for visualization.
   ```
   apt update
   apt install -y ros-noetic-rviz
   ```

In your persistent docker scr file, there should be:  quest2ros  realsense-ros  ROS-TCP-Endpoint  xarm_ros CloudGripper_Manipulation


# Install Teleop
1. Warp robot and quest info to get timestamped topic for sychronization, build teleop_msgs
   ```
   cd ~/catkin_ws
   catkin build teleop_msgs
   source ~/catkin_ws/devel/setup.zsh

   ```

   Test 
   ```
   rosmsg show teleop_msgs/OVR2ROSInputsStamped
   rosmsg show teleop_msgs/RobotMsgStamped
   ```

2. Add scripts to launch nodes for timestamped topics, build cloudgripper_teleop
   ```
   cd ~/catkin_ws
   catkin build cloudgripper_teleop
   source ~/catkin_ws/devel/setup.zsh
   ```


# Run Teleop / Data collection / Policy eval
1. Initialize docker, source
   ```
   sudo docker start -ai ros1_noetic
   ```
2. Start ROS core
   ```
   roscore
   ```
3. Run Teleop
   python src/scripts/run_quest_xarm_teleop_sync.py

4. run data collection
   python src/scripts/run_data_collection.py

5. run policy 
   python src/scripts/run_policy_eval.py