"""Names given with a new text replace the names declared before (Unreleased).

Requirement: when the user's agent changes or corrects a fact and says who the new text is about (`entities`), a person
the fact named before and no longer names must stop being linked to it — otherwise probe/reason by that person keeps
returning a fact about someone else. Without `entities`, the same holds for a declared name only the old text gave;
a declared name no text says stays (a fact about "my teacher" declared as a person keeps that person across a
rewording). Undoing such a change brings the old names back as far as
the texts tell them. Names here are made up."""
from __future__ import annotations

import json
import sqlite3

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.store import MemoryStore

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


# Review of PR #52 (Codex): a revert brings back the names the version had, also those no text gives
@pytest.mark.parametrize("action", ["change", "correct", "update"])
def test_reverting_brings_back_a_one_word_name(p, action):
    fid = _add(p, "Masha teaches piano.", "Masha")
    _call(p, action=action, fact_id=fid, content="Olga teaches piano.", entities=["Olga"])
    _call(p, action="revert", fact_id=fid)
    assert _names(p, fid) == ["Masha"]
    assert "Masha teaches piano." in _probed(p, "Masha")


@pytest.mark.parametrize("action", ["change", "correct", "update"])
def test_reverting_brings_back_a_name_no_text_gives(p, action):
    fid = _add(p, "My teacher plays jazz on Fridays.", "Ben Cole")
    _call(p, action=action, fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    _call(p, action="revert", fact_id=fid)
    assert _names(p, fid) == ["Ben Cole"]
    assert "My teacher plays jazz on Fridays." not in _probed(p, "Pia Lund")


def test_a_reverted_name_someone_deleted_stays_deleted(p):
    fid = _add(p, "My teacher plays jazz on Fridays.", "Ben Cole", "Ada Moss")
    _add(p, "Ben Cole tunes pianos.", "Ben Cole")
    _call(p, action="change", fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    _call(p, action="delete_entity", entity="Ben Cole")
    _call(p, action="revert", fact_id=fid)
    assert _names(p, fid) == ["Ada Moss"]


def test_forget_leaves_no_name_a_version_held(p, tmp_path):
    fid = _add(p, "My teacher plays jazz on Fridays.", "Quill Varga")
    _call(p, action="change", fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    assert _saved(p) == ["Quill Varga"]  # the version's name was kept, so forget has something to remove
    _call(p, action="forget", fact_id=fid)
    p.shutdown()
    data = b"".join(f.read_bytes() for f in tmp_path.iterdir() if f.name.startswith("m.db"))
    assert b"Quill Varga" not in data


# Review of PR #52 (Codex): "still named by the new text" follows the rules of entity identity, aliases included
@pytest.mark.parametrize("first,old,new", [
    ("Straße", "Straße teaches piano.", "strasse teaches drums."),
    ("José", "José teaches piano.", "Jose teaches drums."),
    ("Маша", "Маша учит игре на пианино.", "МАША учит игре на барабанах."),
    ("Семён", "Семён учит игре на пианино.", "Семен учит игре на барабанах."),
], ids=["casefold-sharp-s", "accents", "cyrillic-case", "cyrillic-yo"])
def test_a_name_the_new_text_still_gives_in_another_spelling_stays(p, first, old, new):
    fid = _add(p, old, first)
    _call(p, action="change", fact_id=fid, content=new)
    assert _names(p, fid) == [first]


@pytest.mark.parametrize("alias,new", [("Bobby", "bobby fixes cars now."), ("Großi", "grossi fixes cars now.")],
                         ids=["as-written", "casefold-sharp-s"])
def test_a_name_the_new_text_gives_by_an_alias_stays(p, alias, new):
    fid = _add(p, "Robert Lane fixes bikes.", "Robert Lane")
    _call(p, action="alias", entity="Robert Lane", content=alias)
    _call(p, action="change", fact_id=fid, content=new)
    assert _names(p, fid) == ["Robert Lane"]


# Second review of PR #52 (Codex): what the saved names of a version may and may not bring back
def _open(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": DIM})
    provider.initialize(session_id="s")
    return provider


def _saved(provider):
    return sorted(r[0] for r in provider._store.read("SELECT name FROM version_names WHERE name_key != ''"))


def _file_bytes(tmp_path):
    return b"".join(f.read_bytes() for f in tmp_path.iterdir() if f.name.startswith("m.db"))


def _older_delete_entity(store, name):
    """What delete_entity of 0.20.2 writes (it knows no version_names): the links, the entity row and its aliases go,
    the name gets its tombstones, and the file is rewritten."""
    with store._tx():
        (entity_id,) = store.entity_ids(name)
        store._write("DELETE FROM fact_entities WHERE entity_id = ?", (entity_id,))
        store._drop_entity_row(entity_id)
        for digest in sorted(store._name_digests(name)):
            store._write("INSERT OR IGNORE INTO entity_tombstones (digest) VALUES (?)", (digest,))
        store._set_meta("scrub_pending", "1")
    assert store.scrub()


def test_a_name_an_older_release_deleted_goes_from_the_saved_names_on_the_next_open(tmp_path):
    p = _open(tmp_path)
    fid = _add(p, "My teacher plays jazz on Fridays.", "Quill Varga")
    _add(p, "My neighbour fixes bikes.", "Quill Varga")
    _call(p, action="change", fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    _older_delete_entity(p._store, "Quill Varga")
    p.shutdown()
    p = _open(tmp_path)
    assert _saved(p) == []
    p.shutdown()
    assert b"Quill Varga" not in _file_bytes(tmp_path)
    p = _open(tmp_path)
    try:
        _add(p, "Quill Varga sings in a choir.", "Quill Varga")  # named again: the tombstone is lifted
        _call(p, action="revert", fact_id=fid)
        assert _names(p, fid) == []
    finally:
        p.shutdown()


def test_naming_a_deleted_name_again_does_not_bring_back_what_a_version_saved_of_it(p):
    fid = _add(p, "My teacher plays jazz on Fridays.", "Quill Varga")
    _add(p, "My neighbour fixes bikes.", "Quill Varga")
    _call(p, action="change", fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    _older_delete_entity(p._store, "Quill Varga")  # while this process holds the file: no open in between
    _add(p, "Quill Varga sings in a choir.", "Quill Varga")
    _call(p, action="revert", fact_id=fid)
    assert _names(p, fid) == []


def test_a_version_whose_only_saved_name_was_deleted_brings_back_none(p):
    fid = _add(p, "My teacher plays jazz on Fridays.", "Ben Cole")
    _add(p, "My neighbour fixes bikes.", "Ben Cole")
    _call(p, action="change", fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    _call(p, action="delete_entity", entity="Ben Cole")
    _call(p, action="revert", fact_id=fid)
    assert _names(p, fid) == []


@pytest.mark.parametrize("form", ["entity", "entities"])
def test_delete_entity_finds_a_name_only_a_version_holds(p, tmp_path, form):
    fid = _add(p, "My teacher plays jazz on Fridays.", "Quill Varga")
    _call(p, action="change", fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    args = {"entity": "Quill Varga"} if form == "entity" else {"entities": ["Quill Varga"]}
    out = _call(p, action="delete_entity", **args)
    assert "Quill Varga" in json.dumps(out)
    _call(p, action="revert", fact_id=fid)
    assert _names(p, fid) == []
    p.shutdown()
    assert b"Quill Varga" not in _file_bytes(tmp_path)


def test_saved_names_are_not_used_again_once_their_version_came_back(tmp_path):
    p = _open(tmp_path)
    fid = _add(p, "My teacher plays jazz on Fridays.", "Ben Cole")
    _call(p, action="change", fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    _call(p, action="revert", fact_id=fid)
    _call(p, action="update", fact_id=fid, entities=["Ada Moss"])
    p.shutdown()
    # A release before 0.8.0 (no history) changes the text in place; this one reads that as a change of the fact
    conn = sqlite3.connect(str(tmp_path / "m.db"))
    conn.execute("UPDATE facts SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                 ("My teacher plays rock on Fridays.", fid))
    conn.commit()
    conn.close()
    p = _open(tmp_path)
    try:
        assert _call(p, action="revert", fact_id=fid)["status"] == "reverted"
        assert _names(p, fid) == ["Ada Moss", "Ben Cole"]
    finally:
        p.shutdown()


# Third review of PR #52 (Codex): a deleted name leaves no saved copy, whichever writer deleted it and however spelt
def test_a_name_an_older_release_deletes_is_gone_for_a_second_provider_of_the_process(tmp_path):
    first = _open(tmp_path)
    try:
        fid = _add(first, "My teacher plays jazz on Fridays.", "Quill Varga")
        _add(first, "My neighbour fixes bikes.", "Quill Varga")
        _call(first, action="change", fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
        _older_delete_entity(first._store, "Quill Varga")
        second = _open(tmp_path)  # the same process: it shares the connection the first one migrated
        try:
            assert _saved(second) == []
        finally:
            second.shutdown()
    finally:
        first.shutdown()


def test_naming_a_deleted_name_again_leaves_no_copy_of_what_versions_kept(p, tmp_path):
    fid = _add(p, "My teacher plays jazz on Fridays.", "Quill Varga")
    _add(p, "My neighbour fixes bikes.", "Quill Varga")
    _call(p, action="change", fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    _older_delete_entity(p._store, "Quill Varga")
    _add(p, "QUILL VARGA sings in a choir.", "QUILL VARGA")  # the same name again, spelt otherwise
    assert b"Quill Varga" not in _file_bytes(tmp_path)  # the file and its WAL while the store is open


def test_opening_the_store_does_not_hash_the_saved_names_again(tmp_path, monkeypatch):
    p = _open(tmp_path)
    fid = _add(p, "My teacher plays jazz on Fridays.", "Quill Varga", "Ada Moss")
    _call(p, action="change", fact_id=fid, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    _add(p, "My neighbour fixes bikes.", "Bo Lin")
    _call(p, action="delete_entity", entity="Bo Lin")  # a tombstone exists, none was added since
    p.shutdown()
    hashed = []
    real = MemoryStore.digest
    monkeypatch.setattr(MemoryStore, "digest", lambda self, text: hashed.append(text) or real(self, text))
    p = _open(tmp_path)
    p.shutdown()
    assert not [t for t in hashed if "quill" in t or "ada moss" in t], hashed


@pytest.mark.parametrize("first,second", [("José", "Jose"), ("Łukasz", "Lukasz")], ids=["accent", "polish-l"])
@pytest.mark.parametrize("held", ["history-only", "entity"])
def test_delete_entity_clears_every_spelling_its_tombstone_covers(p, tmp_path, first, second, held):
    a = _add(p, "My teacher plays jazz on Fridays.", first)
    b = _add(p, "My coach runs on Sundays.", second)
    if held == "entity":
        _add(p, "My neighbour fixes bikes.", first)
    _call(p, action="change", fact_id=a, content="My teacher plays blues on Fridays.", entities=["Pia Lund"])
    _call(p, action="change", fact_id=b, content="My coach swims on Sundays.", entities=["Pia Lund"])
    assert _saved(p) == sorted([first, second])
    out = _call(p, action="delete_entity", entity=first)
    assert out["scrub"] == "done"
    assert _saved(p) == []
    p.shutdown()
    data = _file_bytes(tmp_path)
    assert first.encode() not in data and second.encode() not in data
