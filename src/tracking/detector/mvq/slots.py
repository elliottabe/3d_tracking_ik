"""Instance-slot and sex integer codes, shared by inference and training.

>>> SLOT_FEMALE, SEX_FEMALE
(1, 0)
"""

from __future__ import annotations

SLOT_PROMPTED, SLOT_FEMALE, SLOT_MALE, SLOT_OTHER = 0, 1, 2, 3
N_SLOTS = 4

SEX_FEMALE, SEX_MALE, SEX_UNKNOWN = 0, 1, -1
SEX_PRESENT_UNKNOWN = 2
