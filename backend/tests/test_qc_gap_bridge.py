from app.services.video_quality_segment import _bridge_short_gaps, _longest_contiguous_true


def test_bridge_short_gaps_merges_nearby_true_runs() -> None:
    flags = [False, True, True, False, False, True, True, True, False]
    merged = _bridge_short_gaps(flags, max_gap=2)
    seg = _longest_contiguous_true(merged)
    assert seg == (1, 7)


def test_bridge_short_gaps_respects_max_gap() -> None:
    flags = [True, True, False, False, False, True]
    merged = _bridge_short_gaps(flags, max_gap=2)
    seg = _longest_contiguous_true(merged)
    assert seg == (0, 1)
