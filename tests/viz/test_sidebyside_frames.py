"""Which bout frames the side-by-side renders."""

from tracking.viz.sidebyside import _sample_frames


def test_sequence_is_contiguous_from_start_and_capped():
    assert _sample_frames(1000, None, 8, max_frames=300) == list(range(300))
    assert _sample_frames(120, None, 8, max_frames=300) == list(range(120))
    assert _sample_frames(50, None, 8, max_frames=None) == list(range(50))


def test_zero_falls_back_to_preview_and_explicit_frames_win():
    assert len(_sample_frames(1000, None, 8, max_frames=0)) == 8
    assert _sample_frames(1000, [5, 900], 8, max_frames=300) == [5, 900]
