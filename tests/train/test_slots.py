from tracking.detector.mvq import policy, slots


def test_slot_values_are_the_p3a_table():
    assert (slots.SLOT_PROMPTED, slots.SLOT_FEMALE, slots.SLOT_MALE, slots.SLOT_OTHER) == (
        0,
        1,
        2,
        3,
    )
    assert slots.N_SLOTS == 4


def test_sex_codes():
    assert (slots.SEX_FEMALE, slots.SEX_MALE, slots.SEX_UNKNOWN) == (0, 1, -1)
    assert slots.SEX_PRESENT_UNKNOWN == 2


def test_policy_no_longer_defines_the_table_itself():
    import inspect

    src = inspect.getsource(policy)
    assert "SLOT_PROMPTED, SLOT_FEMALE, SLOT_MALE, SLOT_OTHER =" not in src
    assert "N_SLOTS = 4" not in src
    assert policy.SLOT_PROMPTED == slots.SLOT_PROMPTED
    assert policy.N_SLOTS == slots.N_SLOTS


def test_typed_candidates_still_excludes_slot_zero_at_four_slots():
    assert policy.typed_candidates(4) == [1, 2, 3]
    assert policy.typed_candidates(3) == [0, 1, 2]
