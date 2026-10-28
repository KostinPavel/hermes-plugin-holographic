"""Names given with a new text replace the names declared before (Unreleased).

Requirement: when the user's agent changes or corrects a fact and says who the new text is about (`entities`), a person
the fact named before and no longer names must stop being linked to it — otherwise probe/reason by that person keeps
returning a fact about someone else. Without `entities`, the same holds for a declared name only the old text gave;
a declared name no text says stays (a fact about "my teacher" declared as a person keeps that person across a
rewording). Undoing such a change brings the old names back as far as
the texts tell them. Names here are made up."""
from __future__ import annotations

import json

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider

DIM = 64


@pytest.fixture
def p(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": DIM})
    provider.initialize(session_id="s")
    yield provider
    provider.shutdown()


def _call(provider, **args):
    out = json.loads(provider.handle_tool_call("fact_store", args))
    assert "error" not in out, out
    return out


def _add(provider, content, *entities):
    return _call(provider, action="add", content=content, origin="said", entities=list(entities))["fact_id"]


def _probed(provider, name):
    return [r["content"] for r in _call(provider, action="probe", entity=name)["results"]]


def _names(provider, fact_id):
    return sorted(r[0] for r in provider._store.read(
        "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id WHERE fe.fact_id = ?",
        (fact_id,)))


OLD, NEW = "Leo's guitar teacher is Ben Cole.", "Leo's guitar teacher is Omar Reid."


@pytest.mark.parametrize("action", ["change", "correct", "update"])
def test_a_new_text_with_names_is_not_found_under_the_name_it_dropped(p, action):
    fid = _add(p, OLD, "Leo", "Ben Cole")
    _add(p, "Ben Cole holds lessons at the Oak Street studio.", "Ben Cole")  # Ben Cole stays a known person
    _call(p, action=action, fact_id=fid, content=NEW, entities=["Leo", "Omar Reid"])
    assert NEW not in _probed(p, "Ben Cole")
    assert NEW in _probed(p, "Omar Reid")
    assert NEW in _probed(p, "Leo")
    assert _names(p, fid) == ["Leo", "Omar Reid"]


def test_a_name_only_the_dropped_declaration_held_goes_as_an_entity(p):
    fid = _add(p, "My teacher retired last spring.", "Ivo Marsh")
    _call(p, action="change", fact_id=fid, content="My teacher is now Pia Lund.", entities=["Pia Lund"])
    assert _names(p, fid) == ["Pia Lund"]
    assert "Ivo Marsh" not in [e["name"] for e in _call(p, action="entities")["entities"]]


def test_without_names_a_rewording_keeps_who_the_fact_is_about(p):
    fid = _add(p, "My teacher plays in a jazz trio.", "Ben Cole")
    _call(p, action="change", fact_id=fid, content="My teacher left the jazz trio.")
    assert "My teacher left the jazz trio." in _probed(p, "Ben Cole")


def test_names_the_text_still_gives_stay_linked_without_being_declared_again(p):
    fid = _add(p, OLD, "Leo", "Ben Cole")
    _call(p, action="change", fact_id=fid, content="Leo and Ben Cole now practise on Sundays.", entities=["Leo"])
    assert _names(p, fid) == ["Ben Cole", "Leo"]


def test_only_unusable_names_do_not_wipe_the_declared_ones(p):
    fid = _add(p, "My teacher plays in a jazz trio.", "Ben Cole")
    _call(p, action="change", fact_id=fid, content="My teacher left the jazz trio.", entities=["   "])
    assert _names(p, fid) == ["Ben Cole"]


def test_same_text_declarations_still_add(p):
    fid = _add(p, "My teacher plays in a jazz trio.", "Ben Cole")
    _call(p, action="correct", fact_id=fid, content="My teacher plays in a jazz trio.", entities=["Ada Moss"])
    _call(p, action="update", fact_id=fid, entities=["Pia Lund"])
    assert _names(p, fid) == ["Ada Moss", "Ben Cole", "Pia Lund"]


def test_update_with_the_same_text_and_names_adds_them(p):
    fid = _add(p, "My teacher plays in a jazz trio.", "Ben Cole")
    _call(p, action="update", fact_id=fid, content="My teacher plays in a jazz trio.", entities=["Ada Moss"])
    assert _names(p, fid) == ["Ada Moss", "Ben Cole"]


def test_reverting_the_change_restores_the_old_names(p):
    fid = _add(p, OLD, "Leo", "Ben Cole")
    _add(p, "Ben Cole holds lessons at the Oak Street studio.", "Ben Cole")
    _call(p, action="change", fact_id=fid, content=NEW, entities=["Leo", "Omar Reid"])
    assert _call(p, action="revert", fact_id=fid)["status"] == "reverted"
    assert _names(p, fid) == ["Ben Cole", "Leo"]
    assert OLD not in _probed(p, "Omar Reid")
    assert OLD in _probed(p, "Ben Cole")


def test_reverting_a_rewording_keeps_a_name_no_text_gives(p):
    fid = _add(p, "My teacher plays in a jazz trio.", "Ben Cole")
    _call(p, action="change", fact_id=fid, content="My teacher left the jazz trio.")
    _call(p, action="revert", fact_id=fid)
    assert _names(p, fid) == ["Ben Cole"]


@pytest.mark.parametrize("action", ["change", "correct", "update"])
def test_a_new_text_without_names_is_not_found_under_a_name_only_the_old_text_gave(p, action):
    fid = _add(p, OLD, "Leo", "Ben Cole")
    _add(p, "Ben Cole holds lessons at the Oak Street studio.", "Ben Cole")
    _call(p, action=action, fact_id=fid, content=NEW)
    assert NEW not in _probed(p, "Ben Cole")
    assert NEW in _probed(p, "Leo")


@pytest.mark.parametrize("action", ["change", "correct", "update"])
def test_a_name_no_text_gives_survives_a_new_text_without_names(p, action):
    fid = _add(p, "My teacher plays in a jazz trio on Fridays.", "Ben Cole")
    _call(p, action=action, fact_id=fid, content="My teacher plays in a jazz trio on Saturdays.")
    assert _names(p, fid) == ["Ben Cole"]
