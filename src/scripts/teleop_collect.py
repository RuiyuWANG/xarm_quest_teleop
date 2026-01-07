#!/usr/bin/env python3
import argparse, yaml, os
import rospy
from geometry_msgs.msg import Twist

from vr_pipeline.robot.xarm import RobotXArm, XArmServiceNames
from vr_pipeline.vr.quest2 import Quest2Client
from vr_pipeline.vr.teleop import VRTeleopController
from vr_pipeline.sensors.cameras import CameraRig
from vr_pipeline.data.collector import DataCollector

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    xcfg = load_yaml(cfg["robot"]["config"])

    rospy.init_node("teleop_collect_py", anonymous=True)

    # --- Quest msg imports (replace with your actual package) ---
    from quest2ros_msgs.msg import HandInputs, HapticFeedback  # <- change if needed

    quest_cfg = cfg["quest"]
    q = Quest2Client(
        left_pose=quest_cfg["left"]["pose"],
        left_twist=quest_cfg["left"]["twist"],
        left_inputs=quest_cfg["left"]["inputs"],
        left_haptics_pub=quest_cfg["left"]["haptics_pub"],
        right_pose=quest_cfg["right"]["pose"],
        right_twist=quest_cfg["right"]["twist"],
        right_inputs=quest_cfg["right"]["inputs"],
        right_haptics_pub=quest_cfg["right"]["haptics_pub"],
        inputs_msg_type=HandInputs,
        haptics_msg_type=HapticFeedback,
    )

    # Robot services
    srv = XArmServiceNames(
        gripper_move=xcfg["services"]["gripper_move"],
        move_joint=xcfg["services"]["move_joint"],
        move_pose=xcfg["services"]["move_pose"],
        set_ee_twist=xcfg["services"]["set_ee_twist"],
        stop=xcfg["services"]["stop"],
        home=xcfg["services"]["home"],
    )
    robot = RobotXArm(
        srv=srv,
        gripper_min=xcfg["limits"]["gripper_min"],
        gripper_max=xcfg["limits"]["gripper_max"],
    )

    cameras = CameraRig(cfg["cameras"])
    teleop = VRTeleopController(cfg["teleop"])
    collector = DataCollector(cfg["data_root"], cameras, q, control_source="right" if cfg["teleop"]["use_right_hand"] else "left")

    rate_hz = float(cfg.get("rate_hz", 20.0))
    rate = rospy.Rate(rate_hz)

    was_enabled = False
    haptics_sent = False

    while not rospy.is_shutdown():
        hand = q.right if cfg["teleop"]["use_right_hand"] else q.left
        enabled = teleop.enabled(hand.inputs)

        if enabled and not was_enabled:
            # start episode
            meta = {"config": cfg, "xarm_config": xcfg}
            collector.start_episode(meta)
            haptics_sent = False

        if (not enabled) and was_enabled:
            collector.stop_episode()

        if enabled:
            if not haptics_sent and "haptics_on_enable" in cfg["teleop"]:
                hh = cfg["teleop"]["haptics_on_enable"]
                q.send_haptics("right" if cfg["teleop"]["use_right_hand"] else "left",
                               frequency=hh["frequency"], amplitude=hh["amplitude"])
                haptics_sent = True

            # Twist teleop
            if cfg["teleop"]["control_mode"] == "ee_twist":
                cmd_twist = teleop.ee_twist_cmd(hand.twist)
                # send to robot (wire service in RobotXArm.set_ee_twist)
                try:
                    robot.set_ee_twist(cmd_twist)
                except NotImplementedError:
                    pass

            # Gripper
            gp = teleop.gripper_pulse(hand.inputs)
            robot.move_gripper(gp)

            # Write dataset step
            action = {
                "mode": cfg["teleop"]["control_mode"],
                "ee_twist": None if hand.twist is None else {
                    "lin": [cmd_twist.linear.x, cmd_twist.linear.y, cmd_twist.linear.z],
                    "ang": [cmd_twist.angular.x, cmd_twist.angular.y, cmd_twist.angular.z],
                },
                "gripper_pulse": gp,
            }
            collector.write_step(action)

        was_enabled = enabled
        rate.sleep()

if __name__ == "__main__":
    main()
