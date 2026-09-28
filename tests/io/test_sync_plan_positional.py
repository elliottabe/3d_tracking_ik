"""A sync plan whose cameras carry `positions: null` maps slots positionally."""

from tracking.io.video import _slot_position, _slot_present


def test_null_positions_are_positional():
    plan = {"status": "no_drops", "cameras": {"CamA": {"positions": None}}}
    assert _slot_present(plan, "CamA", 1234)
    assert _slot_position(plan, "CamA", 1234) == 1234
