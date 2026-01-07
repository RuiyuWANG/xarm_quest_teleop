#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Interface for obtaining information
"""

import os
import sys
import time
import numpy as np
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

ip = "192.168.1.244"
# ip = "192.168.1.244"
home = np.array(INIT_POSE)
x, y, z, roll, pitch, yaw, gripper_open = home

arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(2)
arm.set_state(state=0)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(mode=0)
arm.set_servo_angle(servo_id=8, angle=[0, -45, 0, 45, 0, 90, 0], is_radian=False)

print('=' * 50)
print('version:', arm.get_version())
print('state:', arm.get_state())
print('cmdnum:', arm.get_cmdnum())
print('err_warn_code:', arm.get_err_warn_code())
print('position(°):', arm.get_position(is_radian=False))
print('position(radian):', arm.get_position(is_radian=True))
print('angles(°):', arm.get_servo_angle(is_radian=False))
print('angles(radian):', arm.get_servo_angle(is_radian=True))
print('angles(°)(servo_id=1):', arm.get_servo_angle(servo_id=1, is_radian=False))
print('angles(radian)(servo_id=1):', arm.get_servo_angle(servo_id=1, is_radian=True))\

arm.disconnect()