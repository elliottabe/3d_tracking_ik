import pytest

from tracking.curate.stages import STAGES, ordered, stage_by_name


def test_declaration_order_is_dependency_order():
    assert [s.name for s in STAGES] == ["merge", "import_masks", "validate", "package"]


def test_ordered_sorts_into_declaration_order():
    assert ordered(["package", "merge", "validate"]) == ["merge", "validate", "package"]


def test_ordered_rejects_unknown_stage():
    with pytest.raises(KeyError, match="nope"):
        ordered(["nope"])


def test_stage_by_name_round_trips():
    assert stage_by_name("merge").name == "merge"
