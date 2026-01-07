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