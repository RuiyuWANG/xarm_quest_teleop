#!/usr/bin/env python3
# scripts/test_quest2_interface.py
"""
Test node for Quest2Interface.

What it tests:
  1) Subscriptions are alive (pose/twist/inputs received)
  2) Getter helpers work: button_lower, press_index, press_middle
  3) Haptics publisher works: vibrate(left/right/both)
  4) Optional: prints twist values while holding button_lower

Run:
  rosrun <your_pkg> test_quest2_interface.py

Optional params:
  ~hand: "right" or "left" (default "right")
  ~rate_hz: print rate (default 20)
  ~haptic_test: bool (default True)
  ~haptic_freq: float (default 120)
  ~haptic_amp: float (default 0.3)
  ~pulse_s: float (default 0.06)
"""

import time
import rospy

from src.vr.quest2 import Quest2Interface


def main():
    rospy.init_node("test_quest2_interface", anonymous=False)

    hand = rospy.get_param("~hand", "right").strip().lower()
    rate_hz = float(rospy.get_param("~rate_hz", 20.0))

    haptic_test = bool(rospy.get_param("~haptic_test", True))
    haptic_freq = float(rospy.get_param("~haptic_freq", 120.0))
    haptic_amp = float(rospy.get_param("~haptic_amp", 0.3))
    pulse_s = float(rospy.get_param("~pulse_s", 0.06))

    q = Quest2Interface(debug=False)

    rospy.loginfo("[test] Waiting for Quest2ROS messages...")
    t0 = time.time()
    timeout_s = 8.0
    r = rospy.Rate(50)

    # Wait until we have at least inputs (pose/twist might be absent if not publishing)
    while not rospy.is_shutdown():
        if q.hand(hand).inputs is not None:
            break
        if time.time() - t0 > timeout_s:
            rospy.logerr(f"[test] Timeout waiting for {hand} inputs. Is Quest2ROS running?")
            return
        r.sleep()

    rospy.loginfo(f"[test] Got {hand} inputs ✅")

    # Haptic sanity pulse
    if haptic_test:
        rospy.loginfo(f"[test] Sending {hand} haptic pulse: freq={haptic_freq} amp={haptic_amp}")
        q.vibrate(hand, haptic_freq, haptic_amp)
        time.sleep(pulse_s)
        q.vibrate(hand, 0.0, 0.0)

        rospy.loginfo("[test] Sending both-hand haptic pulse")
        q.vibrate_both(haptic_freq, haptic_amp)
        time.sleep(pulse_s)
        q.vibrate_both(0.0, 0.0)

    rate = rospy.Rate(rate_hz)
    rospy.loginfo("[test] Printing state. Hold button_lower to see twist streamed + vibration tick.")

    while not rospy.is_shutdown():
        lower = q.button_lower(hand)
        idx = q.press_index(hand)
        mid = q.press_middle(hand)

        h = q.hand(hand)
        tw = h.twist

        if tw is not None:
            lin = (tw.linear.x, tw.linear.y, tw.linear.z)
            ang = (tw.angular.x, tw.angular.y, tw.angular.z)
        else:
            lin = None
            ang = None

        rospy.loginfo_throttle(
            0.25,
            f"[{hand}] lower={lower} index={idx:.3f} middle={mid:.3f} "
            f"twist_lin={lin} twist_ang={ang}"
        )

        # Optional: while holding deadman, send gentle “tick” haptic as feedback
        if lower:
            q.vibrate(hand, frequency=80.0, amplitude=0.1)
        else:
            q.vibrate(hand, frequency=0.0, amplitude=0.0)

        rate.sleep()


if __name__ == "__main__":
    main()
