# tests/conftest.py
import types
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def fake_rospy(monkeypatch):
    """
    A small rospy stub: ServiceProxy, wait_for_service, Subscriber, Publisher,
    Time.now(), sleep, Rate. Good enough for unit tests.
    """
    rospy = types.SimpleNamespace()

    rospy.wait_for_service = MagicMock()
    rospy.ServiceProxy = MagicMock()
    rospy.Subscriber = MagicMock()

    pub = MagicMock()
    pub.publish = MagicMock()
    rospy.Publisher = MagicMock(return_value=pub)

    class _Time:
        @staticmethod
        def now():
            return types.SimpleNamespace(to_sec=lambda: 123.456)

    rospy.Time = _Time
    rospy.sleep = MagicMock()
    rospy.Rate = MagicMock(return_value=types.SimpleNamespace(sleep=MagicMock()))
    rospy.is_shutdown = MagicMock(return_value=False)

    rospy.loginfo = MagicMock()
    rospy.logwarn = MagicMock()
    rospy.logerr = MagicMock()
    rospy.set_param = MagicMock()
    rospy.get_param = MagicMock()

    monkeypatch.setitem(__import__("sys").modules, "rospy", rospy)
    return rospy


@pytest.fixture
def fake_message_filters(monkeypatch):
    """
    Stubs message_filters.Subscriber and ApproximateTimeSynchronizer.
    """
    mf = types.SimpleNamespace()

    class _Subscriber:
        def __init__(self, topic, msg_type):
            self.topic = topic
            self.msg_type = msg_type

    class _ATS:
        def __init__(self, subs, queue_size, slop):
            self.subs = subs
            self.queue_size = queue_size
            self.slop = slop
            self._cb = None

        def registerCallback(self, cb):
            self._cb = cb

        # helper for tests: trigger callback manually
        def _fire(self, *msgs):
            assert self._cb is not None
            self._cb(*msgs)

    mf.Subscriber = _Subscriber
    mf.ApproximateTimeSynchronizer = _ATS

    monkeypatch.setitem(__import__("sys").modules, "message_filters", mf)
    return mf
