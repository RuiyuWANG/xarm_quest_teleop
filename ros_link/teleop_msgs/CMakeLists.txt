cmake_minimum_required(VERSION 3.0.2)
project(teleop_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  message_generation
  quest2ros
  xarm_msgs
)

add_message_files(
  FILES
  OVR2ROSInputsStamped.msg
  RobotMsgStamped.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  quest2ros
  xarm_msgs
)

catkin_package(
  CATKIN_DEPENDS
    std_msgs
    geometry_msgs
    message_runtime
    quest2ros
    xarm_msgs
)
