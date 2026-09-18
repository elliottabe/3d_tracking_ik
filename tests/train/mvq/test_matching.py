import jax.numpy as jnp
import numpy as np

from tracking.detector.mvq.slots import (
    N_SLOTS,  # noqa: F401 - documents the default n_instances assign_slots/slot_ignore use
    SEX_FEMALE,
    SEX_MALE,
    SEX_PRESENT_UNKNOWN,
    SEX_UNKNOWN,
    SLOT_FEMALE,
    SLOT_MALE,
    SLOT_OTHER,
    SLOT_PROMPTED,
)
from tracking.train.mvq.matching import assign_slots, slot_ignore


def _call(sex, valid, on, dist):
    return assign_slots(jnp.asarray(sex, jnp.int8), jnp.asarray(valid, bool),
                        jnp.asarray(on, bool), jnp.asarray(dist, jnp.float32))


def test_a_female_and_a_male_take_their_typed_slots():
    assign, target = _call([[SEX_FEMALE, SEX_MALE]], [[True, True]], [False], [[0.0, 5.0]])
    assert list(np.asarray(assign)[0]) == [SLOT_FEMALE, SLOT_MALE]
    assert list(np.asarray(target)[0]) == [False, True, True, False]


def test_prompt_on_puts_the_host_in_the_prompted_slot():
    assign, _ = _call([[SEX_FEMALE, SEX_MALE]], [[True, True]], [True], [[0.0, 5.0]])
    assert np.asarray(assign)[0, 0] == SLOT_PROMPTED


def test_two_same_sex_flies_put_the_second_in_other():
    assign, _ = _call([[SEX_FEMALE, SEX_FEMALE]], [[True, True]], [False], [[0.0, 5.0]])
    assert sorted(np.asarray(assign)[0].tolist()) == sorted([SLOT_FEMALE, SLOT_OTHER])


def test_an_invalid_fly_is_assigned_minus_one():
    assign, target = _call([[SEX_FEMALE, SEX_MALE]], [[True, False]], [False], [[0.0, 5.0]])
    assert np.asarray(assign)[0, 1] == -1
    assert not np.asarray(target)[0, SLOT_MALE]


def test_unknown_sex_lands_in_other():
    assign, _ = _call([[SEX_UNKNOWN]], [[True]], [False], [[0.0]])
    assert np.asarray(assign)[0, 0] == SLOT_OTHER


def test_slot_ignore_unknown_sex_ignores_every_slot_but_prompted():
    ig = np.asarray(slot_ignore(jnp.asarray([SEX_PRESENT_UNKNOWN], jnp.int8)))[0]
    assert not ig[SLOT_PROMPTED] and ig[SLOT_FEMALE] and ig[SLOT_MALE] and ig[SLOT_OTHER]


def test_slot_ignore_a_named_sex_ignores_that_slot_and_other_only():
    ig = np.asarray(slot_ignore(jnp.asarray([SEX_FEMALE], jnp.int8)))[0]
    assert list(ig) == [False, True, False, True]


def test_slot_ignore_minus_one_ignores_nothing():
    assert not np.asarray(slot_ignore(jnp.asarray([SEX_UNKNOWN], jnp.int8)))[0].any()


def test_matching_does_not_redefine_the_slot_table():
    import tracking.detector.mvq.slots as shared
    import tracking.train.mvq.matching as m
    assert m.N_SLOTS is shared.N_SLOTS
    src = open(m.__file__).read()
    assert "SLOT_PROMPTED, SLOT_FEMALE" not in src, "slot constants must be imported, not redefined"
