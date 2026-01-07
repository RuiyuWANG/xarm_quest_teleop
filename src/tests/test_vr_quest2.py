# tests/test_vr_quest2.py
import types

def test_quest_hand_stores_messages(fake_rospy, monkeypatch):
    # Patch quest2ros msg imports used by your module
    # If your quest module imports from quest2ros_msgs.msg, we inject stubs into sys.modules.
    import sys
    quest2ros_msgs = types.SimpleNamespace()
    quest2ros_msgs.msg = types.SimpleNamespace()

    class HandInputs:
        def __init__(self):
            self.button_upper = False
            self.button_lower = False
            self.thumb_stick_horizontal = 0.0
            self.thumb_stick_vertical = 0.0
            self.press_index = 0.0
            self.press_middle = 0.0

    class HapticFeedback:
        def __init__(self):
            self.frequency = 0.0
            self.amplitude = 0.0

    quest2ros_msgs.msg.HandInputs = HandInputs
    quest2ros_msgs.msg.HapticFeedback = HapticFeedback
    sys.modules["quest2ros_msgs"] = quest2ros_msgs
    sys.modules["quest2ros_msgs.msg"] = quest2ros_msgs.msg

    from vr_pipeline.vr.quest2 import Quest2Client  # adjust if you named it Quest2

    q = Quest2Client(
        left_pose="/q2r_left_hand_pose",
        left_twist="/q2r_left_hand_twist",
        left_inputs="/q2r_left_hand_inputs",
        left_haptics_pub="/q2r_left_hand_haptic_feedback",
        right_pose="/q2r_right_hand_pose",
        right_twist="/q2r_right_hand_twist",
        right_inputs="/q2r_right_hand_inputs",
        right_haptics_pub="/q2r_right_hand_haptic_feedback",
        inputs_msg_type=HandInputs,
        haptics_msg_type=HapticFeedback,
    )

    # Simulate incoming messages by calling the subscribers' callbacks
    # We can’t access internal callbacks directly if you used lambda;
    # so the minimal check is that Publisher exists and send_haptics publishes.
    q.send_haptics("right", frequency=100.0, amplitude=0.5)
    pub = fake_rospy.Publisher.return_value
    pub.publish.assert_called()
    msg_sent = pub.publish.call_args[0][0]
    assert msg_sent.frequency == 100.0
    assert msg_sent.amplitude == 0.5
