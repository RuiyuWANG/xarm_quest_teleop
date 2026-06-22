cmake_minimum_required(VERSION 3.8)
project(xarm_quest_teleop_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(quest2ros REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/OVR2ROSInputsStamped.msg"
  DEPENDENCIES quest2ros std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
