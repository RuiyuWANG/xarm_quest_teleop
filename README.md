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

3. Setup cameras.

   pass

# ROS SET-up
1. install ROS in docker
   source:
      ```
         source /opt/ros/noetic/setup.zsh 2>/dev/null || source /opt/ros/noetic/setup.bash
         source ~/catkin_ws/devel/setup.zsh 2>/dev/null || source ~/catkin_ws/devel/setup.bash
      ```

2. Quest
   Follow:
   ```
      roslaunch ros_tcp_endpoint endpoint.launch tcp_ip:=192.168.0.182 tcp_port:=10000
   ```

   ```
      rosrun quest2ros ros2quest.py
   ```

3. Robot
   Follow:

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

