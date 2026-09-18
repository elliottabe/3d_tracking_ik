"""Label-driven instance-slot assignment (P3a spec §3-4). No prediction enters:
a slot's existence/sex target is a deterministic function of the labels and
the prompt flag, so the existence head never sees coin-flip targets."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from tracking.detector.mvq.slots import (
    N_SLOTS,
    SEX_FEMALE,
    SEX_MALE,
    SEX_PRESENT_UNKNOWN,
    SLOT_FEMALE,
    SLOT_MALE,
    SLOT_OTHER,
    SLOT_PROMPTED,
)


def _assign_one(sex, valid, on, dist, n_instances):
    """One sample. sex (F,) int8, valid (F,) bool, on () bool, dist (F,) float."""
    F = sex.shape[0]
    # host (fly 0) first, then the others by increasing distance from the ROI origin
    order = jnp.argsort(jnp.where(jnp.arange(F) == 0, -jnp.inf, dist))
    typed_all = jnp.where(sex == SEX_FEMALE, SLOT_FEMALE,
                           jnp.where(sex == SEX_MALE, SLOT_MALE, SLOT_OTHER))

    def body(carry, f):
        taken, assign = carry
        typed = typed_all[f]
        slot = jnp.where((f == 0) & on, SLOT_PROMPTED, jnp.where(taken[typed], SLOT_OTHER, typed))
        slot = jnp.where((slot >= 0) & taken[jnp.maximum(slot, 0)], -1, slot)
        slot = jnp.where(valid[f], slot, -1)
        taken = jnp.where(valid[f] & (slot >= 0),
                          taken.at[jnp.maximum(slot, 0)].set(True) | taken, taken)
        return (taken, assign.at[f].set(slot)), None

    init = (jnp.zeros((n_instances,), bool), jnp.full((F,), -1, jnp.int32))
    (taken, assign), _ = jax.lax.scan(body, init, order)
    return assign, taken


def assign_slots(fly_sex, fly_valid, prompt_on, dist, n_instances=N_SLOTS):
    """fly_sex (B,F) int8 {0 F, 1 M, -1 unknown}; fly_valid (B,F); prompt_on (B,);
    dist (B,F) distance of each fly's labelled-3D centroid from the ROI origin.
    Returns assign (B,F) int32 slot per fly (-1 = invalid fly) and
    slot_target (B,I) bool = a fly was assigned to that slot. F <= 2 is assumed
    (slot 3 can hold one fly). A fly whose resolved slot is already taken is
    dropped (assigned -1) rather than colliding with another fly."""
    def one(s, v, o, d):
        return _assign_one(s, v, o, d, n_instances)

    return jax.vmap(one)(fly_sex, fly_valid, prompt_on, dist)


def slot_ignore(unlabelled_sex, n_instances=N_SLOTS):
    """(B,) int8 -> (B,I) bool: slots that get NO existence loss because an
    unlabelled fly present in the window could legitimately occupy them.
    -1: none; 0/1: that sex's slot and OTHER; 2: every slot but PROMPTED."""
    u = unlabelled_sex[:, None]
    s = jnp.arange(n_instances)[None, :]
    other = s == SLOT_OTHER
    return (((u == SEX_FEMALE) & ((s == SLOT_FEMALE) | other))
            | ((u == SEX_MALE) & ((s == SLOT_MALE) | other))
            | ((u == SEX_PRESENT_UNKNOWN) & (s != SLOT_PROMPTED)))
