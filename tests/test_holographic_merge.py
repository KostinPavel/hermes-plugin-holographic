"""0.19.0: `fact_store merge` — two facts that say the same thing become one, without losing either history.
plans/2026-10-25-fact-merge.md (v3) is the contract these tests check.

Requirements, in the user's words:
- a duplicate merged into the fact that stays leaves search, recall and the list; its versions, sources and
  confirmations stay under its own id, and `history` of the keep shows them;
- the text of a merged copy does not come back as a new fact: `add`, auto-extraction, `change`, `update` and a resume
  answer with the keep;
- what held on a date before the merge is still answered, and the copy stops "holding" on the day it was merged;
- the keep keeps its own trust; the group's counts add up once; keywords are joined or replaced as asked;
- one person confirming the same claim twice (once on the copy) raises trust once;
- `forget` of the keep forgets the copies too, a tombstone each; `forget` of a copy leaves the keep as it is;
- an error anywhere in a merge leaves nothing written; the file keeps working after a failed commit;
- a scheduled job merges only what it stored in this session (contract addition of 2026-10-26);
- a file without merges keeps history schema 1; the first merge raises it to 2."""

from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.history import ChangedSince, HistoryMixin, NotFound
from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore

DIM = 256


class Clock:
    def __init__(self, day: str):
        self.day = day


@pytest.fixture
def clock(monkeypatch):
    c = Clock("2026-10-12")
    monkeypatch.setattr(HistoryMixin, "_now", staticmethod(lambda: f"{c.day} 12:00:00"))
    monkeypatch.setattr(HistoryMixin, "_today", lambda self: c.day)
    return c


def _provider(tmp_path, config=None, **init):
    cfg = {"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": DIM, **(config or {})}
    p = HolographicMemoryProvider(config=cfg)
    p.initialize(session_id=init.pop("session_id", "s1"), **init)
    return p


@pytest.fixture
def provider(tmp_path, clock):
    p = _provider(tmp_path)
    yield p
    p.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _feedback(provider, **args):
    return json.loads(provider.handle_tool_call("fact_feedback", args))


def _found(provider, query, **kw):
    return [r["fact_id"] for r in _call(provider, action="search", query=query, min_trust=0, **kw)["results"]]


def _snapshot(store) -> dict:
    """Everything a refused merge must leave as it was."""
    tables = ("facts", "fact_versions", "fact_entities", "entities", "fact_confirmations", "fact_merges", "meta")
    return {t: [tuple(r) for r in store.read(f"SELECT * FROM {t}")] for t in tables}


def _pair(provider):
    """Two facts that say the same, each with a confirmation of its own person, keywords and entities."""
    keep = _call(provider, action="add", content="Alice Smith lives in Paris near the Louvre", origin="said",
                 keywords=["home", "residence"])["fact_id"]
    copy = _call(provider, action="add", content="Alice Smith is living in Paris by the Louvre zorblat",
                 origin="said", keywords=["residence", "flat", "квартира"])["fact_id"]
    return keep, copy


def _raw_bytes(db_path) -> bytes:
    data = b""
    for suffix in ("", "-wal"):
        try:
            with open(f"{db_path}{suffix}", "rb") as fh:
                data += fh.read()
        except FileNotFoundError:
            pass
    return data.lower()


# --- the merge itself ------------------------------------------------------------------------------------------------


def test_merge_two_keeps_one_fact_and_the_copy_leaves_search_recall_and_the_list(provider):
    keep, copy = _pair(provider)
    assert sorted(_found(provider, "Alice Louvre")) == [keep, copy]
    result = _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    assert result["merged"] == [copy] and result["into"] == keep and "error" not in result
    assert _found(provider, "Alice Louvre") == [keep]
    assert _found(provider, "zorblat") == []  # a word only the copy had finds nothing current
    assert [f["fact_id"] for f in _call(provider, action="list")["facts"]] == [keep]
    assert "zorblat" not in provider.prefetch("Where is Alice Smith living, by the Louvre zorblat?")
    assert _call(provider, action="probe", entity="Alice Smith")["count"] == 1
    store = provider._store
    assert store.read("SELECT status FROM fact_versions WHERE fact_id = ?", (copy,))[0][0] == "merged"
    row = store.read("SELECT into_fact_id, into_version_id FROM fact_merges WHERE from_fact_id = ?", (copy,))[0]
    assert row[0] == keep and row[1] == store._current_version(keep)["version_id"]
    assert store.read("SELECT COUNT(*) FROM facts WHERE fact_id = ?", (copy,))[0][0] == 0


def test_merge_three_at_once(provider):
    keep, copy = _pair(provider)
    third = _call(provider, action="add", content="Alice Smith resides in Paris at the Louvre quxxle")["fact_id"]
    result = _call(provider, action="merge", fact_id=keep, **{"from": [copy, third]})
    assert result["merged"] == [copy, third]
    assert [f["fact_id"] for f in _call(provider, action="list")["facts"]] == [keep]
    assert _call(provider, action="history", fact_id=keep)["merged_from"][0]["fact_id"] == copy
    assert [m["fact_id"] for m in _call(provider, action="history", fact_id=keep)["merged_from"]] == [copy, third]


def test_history_of_the_keep_shows_the_copies_with_their_sources_and_confirmations(tmp_path, clock):
    first = _provider(tmp_path, session_id="s1")
    keep, copy = _pair(first)
    first.shutdown()
    later = _provider(tmp_path, session_id="s2", user_id="u2", user_name="Bob")
    _feedback(later, action="confirmed", fact_id=copy)  # Bob confirmed the copy
    result = _call(later, action="merge", fact_id=keep, **{"from": [copy]})
    assert result["merged_confirmations"] == 1
    history = _call(later, action="history", fact_id=keep)
    assert history["current"] is True and "merged_into" not in history
    assert history["merged_confirmations"] == 1
    assert len(history["merged_from"]) == 1
    merged = history["merged_from"][0]
    assert merged["fact_id"] == copy and merged["merged_at"] == "2026-10-12 12:00:00"
    assert [v["status"] for v in merged["versions"]] == ["merged"]
    assert merged["versions"][0]["content"].endswith("zorblat")
    assert merged["versions"][0]["source"]["session_id"] == "s1" and merged["versions"][0]["source"]["origin"] == "said"
    assert all("version_id" in v for v in history["versions"] + merged["versions"])
    of_copy = _call(later, action="history", fact_id=copy)
    assert of_copy["merged_into"] == keep and of_copy["current"] is False
    assert [v["status"] for v in of_copy["versions"]] == ["merged"]
    later.shutdown()


def test_keywords_are_joined_the_keeps_first_or_replaced_or_cleared(provider):
    keep, copy = _pair(provider)
    result = _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    assert result["keywords"] == ["home", "residence", "flat", "квартира"]  # the keep's first, no repeats
    assert "keywords_dropped" not in result
    stored = provider._store.fact_tags(keep)
    assert stored.endswith("keywords: home, residence, flat, квартира")
    assert _found(provider, "квартира") == [keep]  # the copy's keyword now finds the keep
    third = _call(provider, action="add", content="Alice Smith is a Paris resident at the Louvre")["fact_id"]
    result = _call(provider, action="merge", fact_id=keep, **{"from": [third]}, keywords=["dwelling", "address"])
    assert result["keywords"] == ["dwelling", "address"]
    assert provider._store.fact_tags(keep).endswith("keywords: dwelling, address")
    fourth = _call(provider, action="add", content="Alice Smith has her home in Paris at the Louvre")["fact_id"]
    result = _call(provider, action="merge", fact_id=keep, **{"from": [fourth]}, keywords=[])
    assert result["keywords"] == [] and provider._store.fact_tags(keep) == ""
    # the version row follows (update does the same)
    assert provider._store._current_version(keep)["tags"] == ""


def test_keywords_past_the_limit_are_reported_before_the_cut(provider):
    keep = _call(provider, action="add", content="Mom's birthday is May 3", keywords=[f"k{i}" for i in range(8)])["fact_id"]
    copy = _call(provider, action="add", content="Mom was born on May 3", keywords=[f"c{i}" for i in range(8)])["fact_id"]
    result = _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    assert result["keywords"] == [f"k{i}" for i in range(8)] + ["c0", "c1", "c2", "c3"]
    assert "#13" in result["keywords_dropped"] and "limit of 12" in result["keywords_dropped"]


def test_counts_add_up_once_trust_stays_the_keeps_and_the_same_merge_again_changes_nothing(provider):
    keep, copy = _pair(provider)
    store = provider._store
    store.record_retrievals([keep, copy, copy]) and None
    store._write("UPDATE facts SET retrieval_count = 3, helpful_count = 2 WHERE fact_id = ?", (copy,))
    store._write("UPDATE facts SET retrieval_count = 1, helpful_count = 0, trust_score = 0.4 WHERE fact_id = ?", (keep,))
    store._write("UPDATE facts SET trust_score = 0.9 WHERE fact_id = ?", (copy,))
    result = _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    assert result["counts_added"] == {"retrieval": 3, "helpful": 2}
    row = store.read("SELECT retrieval_count, helpful_count, trust_score FROM facts WHERE fact_id = ?", (keep,))[0]
    assert tuple(row) == (4, 2, 0.4)  # the group's past added, the keep's own trust (not the copy's 0.9)
    again = _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    assert again["merged"] == [] and again["already"] == [copy] and "error" not in again
    assert again["counts_added"] == {"retrieval": 0, "helpful": 0}
    row = store.read("SELECT retrieval_count, helpful_count FROM facts WHERE fact_id = ?", (keep,))[0]
    assert tuple(row) == (4, 2)
    assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 1


def test_the_keep_keeps_its_origin_category_and_user_tags(provider):
    keep = _call(provider, action="add", content="Alice Smith lives in Paris near the Louvre", origin="inferred",
                 category="user_pref", tags="family")["fact_id"]
    copy = _call(provider, action="add", content="Alice Smith is living in Paris by the Louvre", origin="said",
                 category="general", keywords=["flat"])["fact_id"]
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    fact = provider._store.read("SELECT category, tags FROM facts WHERE fact_id = ?", (keep,))[0]
    assert fact[0] == "user_pref" and fact[1] == "family\nkeywords: flat"
    assert provider._store.origins([keep])[keep]["origin"] == "inferred"  # the copy's "said" does not rewrite it


# --- the copy's text does not come back ---------------------------------------------------------------------------------


def test_add_of_the_copys_text_answers_with_the_keep(tmp_path, clock):
    first = _provider(tmp_path, session_id="s1")
    keep, copy = _pair(first)
    _call(first, action="merge", fact_id=keep, **{"from": [copy]})
    first.shutdown()
    later = _provider(tmp_path, session_id="s2", user_id="u9", user_name="Nina")
    result = _call(later, action="add", content="Alice Smith is living in Paris by the Louvre zorblat", origin="said",
                   entities=["Louvre"], keywords=["x"])
    assert result["status"] == "exists" and result["fact_id"] == keep and result["merged_from"] == copy
    assert result["confirmations"] == 1  # the person said the same claim again in a later session: the keep's
    assert "keywords_not_set" in result
    assert later._store.count_facts() == 1
    assert later._store.read("SELECT COUNT(*) FROM fact_confirmations WHERE fact_id = ?", (keep,))[0][0] == 1
    names = {r[0] for r in later._store.read(
        "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id WHERE fe.fact_id = ?", (keep,))}
    assert "Louvre" in names  # names declared with it count for the keep
    later.shutdown()


def test_a_scheduled_job_adding_the_copys_text_gets_the_keep_not_a_new_fact(tmp_path, clock):
    main = _provider(tmp_path)
    keep, copy = _pair(main)
    _call(main, action="merge", fact_id=keep, **{"from": [copy]})
    main.shutdown()
    cron = _provider(tmp_path, agent_context="cron", platform="cron", session_id="cron1")
    result = _call(cron, action="add", content="Alice Smith is living in Paris by the Louvre zorblat")
    assert result["status"] == "exists" and result["fact_id"] == keep and cron._store.count_facts() == 1
    cron.shutdown()


def test_auto_extraction_skips_the_copys_text(tmp_path, clock):
    provider = _provider(tmp_path, config={"auto_extract": "true"})
    keep = _call(provider, action="add", content="I prefer green tea in the morning")["fact_id"]
    copy = _call(provider, action="add", content="I prefer green tea every morning zorblat")["fact_id"]
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    provider.on_session_end([{"role": "user", "content": "I prefer green tea every morning zorblat"}])
    assert provider._store.count_facts() == 1
    provider.shutdown()


def test_an_earlier_changed_version_of_the_copy_is_a_past_version_and_a_corrected_one_is_nothing(provider):
    keep = _call(provider, action="add", content="Alice Smith lives in Paris")["fact_id"]
    copy = _call(provider, action="add", content="Alice Smith lived in Lyon")["fact_id"]
    _call(provider, action="change", fact_id=copy, content="Alice Smith is living in Paris")
    _call(provider, action="correct", fact_id=copy, content="Alice Smith is living in Paris, France")
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    past = _call(provider, action="add", content="Alice Smith lived in Lyon")
    assert past["status"] == "past_version" and past["fact_id"] == copy
    gone = _call(provider, action="add", content="Alice Smith is living in Paris")  # corrected: it never held
    assert gone["status"] == "added"
    head = _call(provider, action="add", content="Alice Smith is living in Paris, France")
    assert head["status"] == "exists" and head["fact_id"] == keep


def test_change_update_and_resume_into_the_copys_text_answer_with_the_keep(provider):
    keep, copy = _pair(provider)
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    other = _call(provider, action="add", content="Alice Smith works at the Louvre")["fact_id"]
    changed = _call(provider, action="change", fact_id=other, content="Alice Smith is living in Paris by the Louvre zorblat")
    assert changed == {"fact_id": keep, "status": "exists", "ended": other}
    assert provider._store.count_facts() == 1
    third = _call(provider, action="add", content="Alice Smith paints at the Louvre")["fact_id"]
    updated = _call(provider, action="update", fact_id=third, content="Alice Smith is living in Paris by the Louvre zorblat")
    assert updated["updated"] is False and updated["duplicate_of"] == keep
    resumed = _call(provider, action="change", fact_id=other, content="Alice Smith is living in Paris by the Louvre zorblat")
    assert resumed["status"] == "exists" and resumed["fact_id"] == keep  # the ended fact does not come back as a twin
    assert provider._store.count_facts() == 2


def test_the_keep_may_take_its_copys_wording_back(provider):
    keep, copy = _pair(provider)
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    changed = _call(provider, action="change", fact_id=keep, content="Alice Smith is living in Paris by the Louvre zorblat")
    assert changed["status"] == "changed" and changed["fact_id"] == keep


def test_the_write_check_points_a_near_duplicate_at_the_keep(provider):
    keep, copy = _pair(provider)
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    result = _call(provider, action="add", content="Alice Smith is living in Paris by the Louvre zorblat, she says")
    merged = [p for p in result.get("past_versions", []) if p["fact_id"] == copy]
    assert merged and merged[0]["status"] == "merged" and merged[0]["merged_into"] == keep and "until" not in merged[0]


def test_add_with_a_similar_fact_suggests_merge(provider):
    old = _call(provider, action="add", content="Alice Smith lives in Paris near the Louvre", origin="said")["fact_id"]
    result = _call(provider, action="add", content="Alice Smith lives in Paris near the Louvre museum", origin="said")
    assert result["similar"][0]["fact_id"] == old
    assert f"merge {{fact_id: <old id>, from: [{result['fact_id']}]}}" in result["hint"]
    assert "confirm the old one" not in result["hint"]


# --- as_of -----------------------------------------------------------------------------------------------------------


def test_as_of_before_the_merge_shows_the_copy_under_its_own_id_and_after_only_the_keep(provider, clock):
    keep = _call(provider, action="add", content="Alice Smith lives in Paris", valid_from="2026-01-01")["fact_id"]
    copy = _call(provider, action="add", content="Alice Smith is living in Paris zorblat", valid_from="2026-03-01")["fact_id"]
    clock.day = "2026-10-12"
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    before = _call(provider, action="list", as_of="2026-06-01")["facts"]
    assert {f["fact_id"]: f.get("merged_into") for f in before} == {keep: None, copy: keep}
    copy_entry = next(f for f in before if f["fact_id"] == copy)
    assert copy_entry["status"] == "merged" and copy_entry["valid_to"] == "2026-10-12"
    searched = _call(provider, action="search", query="Alice Paris", as_of="2026-06-01")["results"]
    assert {f["fact_id"]: f.get("merged_into") for f in searched} == {keep: None, copy: keep}
    today = _call(provider, action="list", as_of="2026-10-12")["facts"]
    assert [f["fact_id"] for f in today] == [keep]
    assert [f["fact_id"] for f in _call(provider, action="search", query="Alice Paris", as_of="2026-10-12")["results"]] == [keep]


def test_after_the_keep_changes_the_copy_does_not_hold_the_old_claim_on(provider, clock):
    keep = _call(provider, action="add", content="Alice Smith lives in Paris", valid_from="2026-01-01")["fact_id"]
    copy = _call(provider, action="add", content="Alice Smith is living in Paris zorblat", valid_from="2026-01-01")["fact_id"]
    clock.day = "2026-10-12"
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    clock.day = "2026-12-05"
    _call(provider, action="change", fact_id=keep, content="Alice Smith lives in Berlin", valid_from="2026-12-01")
    after = _call(provider, action="list", as_of="2026-12-03")
    assert [(f["fact_id"], f["content"]) for f in after["facts"]] == [(keep, "Alice Smith lives in Berlin")]
    assert after["undetermined"] == []
    searched = _call(provider, action="search", query="Alice Smith lives", as_of="2026-12-03")["results"]
    assert [(f["fact_id"], f["content"]) for f in searched] == [(keep, "Alice Smith lives in Berlin")]


def test_after_the_keep_ends_the_copy_does_not_hold_on(provider, clock):
    keep = _call(provider, action="add", content="Alice Smith lives in Paris", valid_from="2026-01-01")["fact_id"]
    copy = _call(provider, action="add", content="Alice Smith is living in Paris zorblat", valid_from="2026-01-01")["fact_id"]
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    clock.day = "2026-11-20"
    _call(provider, action="end", fact_id=keep, valid_to="2026-11-01")
    assert _call(provider, action="list", as_of="2026-11-20")["facts"] == []
    assert _call(provider, action="search", query="Alice Paris", as_of="2026-11-20")["results"] == []
    assert [f["fact_id"] for f in _call(provider, action="list", as_of="2026-10-01")["facts"]] == [keep, copy]


# --- confirmations -------------------------------------------------------------------------------------------------------


def test_a_person_who_confirmed_the_copy_does_not_raise_the_keeps_trust_again(tmp_path, clock):
    first = _provider(tmp_path, session_id="s1")
    keep, copy = _pair(first)
    first.shutdown()
    bob = _provider(tmp_path, session_id="s2", user_id="bob")
    assert _feedback(bob, action="confirmed", fact_id=copy)["trust_score"] == pytest.approx(0.65)
    _call(bob, action="merge", fact_id=keep, **{"from": [copy]})
    trust = bob._store.read("SELECT trust_score FROM facts WHERE fact_id = ?", (keep,))[0][0]
    bob.shutdown()
    bob_again = _provider(tmp_path, session_id="s3", user_id="bob")
    result = _feedback(bob_again, action="confirmed", fact_id=keep)
    assert result["confirmed"] is True and result["trust_score"] == pytest.approx(trust)  # known: no second rise
    assert result["confirmations"] == 1  # the keep's own rows
    bob_again.shutdown()
    same_session = _provider(tmp_path, session_id="s2", user_id="carol")  # the copy's session: once per session
    result = _feedback(same_session, action="confirmed", fact_id=keep)
    assert result["confirmed"] is False
    same_session.shutdown()


def test_the_copys_confirmations_do_not_make_the_keep_confirmed(tmp_path, clock):
    first = _provider(tmp_path, session_id="s1")
    keep = _call(first, action="add", content="Alice Smith lives in Paris", origin="inferred")["fact_id"]
    copy = _call(first, action="add", content="Alice Smith is living in Paris zorblat", origin="inferred")["fact_id"]
    first.shutdown()
    bob = _provider(tmp_path, session_id="s2", user_id="bob")
    _feedback(bob, action="confirmed", fact_id=copy)
    _call(bob, action="merge", fact_id=keep, **{"from": [copy]})
    assert bob._store.origins([keep])[keep] == {"origin": "inferred", "confirmations": 0}
    stats = _call(bob, action="stats")
    assert stats["origin"]["confirmations"] == 0 and stats["origin"]["inferred"] == 1
    assert _call(bob, action="history", fact_id=keep)["merged_confirmations"] == 1
    bob.shutdown()


# --- entities --------------------------------------------------------------------------------------------------------------


def test_the_copys_names_move_to_the_keep_and_its_text_aliases_go(provider):
    keep = _call(provider, action="add", content="Alice Smith lives in Paris")["fact_id"]
    copy = _call(provider, action="add", content="Alice Smith aka Ally Smith lives in Paris with Bob Brown",
                 entities=["Masha"])["fact_id"]
    store = provider._store
    assert store.read("SELECT source FROM entity_aliases WHERE alias = 'Ally Smith'")[0][0] == "aka"
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    names = {r[0] for r in store.read(
        "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id WHERE fe.fact_id = ?", (keep,))}
    assert {"Alice Smith", "Bob Brown", "Masha"} <= names
    assert store.read("SELECT entity_type FROM entities WHERE name = 'Masha'")[0][0] == "declared"
    assert _call(provider, action="probe", entity="Bob Brown")["results"][0]["fact_id"] == keep
    # documented: an alias only the copy's text gave goes with its row; the text stays in the copy's history
    assert store.read("SELECT COUNT(*) FROM entity_aliases WHERE alias = 'Ally Smith'")[0][0] == 0
    assert "Ally Smith" in _call(provider, action="history", fact_id=keep)["merged_from"][0]["versions"][0]["content"]
    assert store.read("SELECT COUNT(*) FROM fact_entities WHERE fact_id = ?", (copy,))[0][0] == 0


# --- counts, chains, export --------------------------------------------------------------------------------------------


def test_history_counts_tell_merged_facts_from_ended_ones(provider):
    keep = _call(provider, action="add", content="Alice Smith lives in Paris")["fact_id"]
    copy = _call(provider, action="add", content="Alice Smith is living in Paris zorblat")["fact_id"]
    _call(provider, action="end", fact_id=copy)
    assert _call(provider, action="stats")["history"]["ended_facts"] == 1
    _call(provider, action="change", fact_id=copy, content="Alice Smith is living in Paris zorblat")  # resumed
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    counts = _call(provider, action="stats")["history"]
    assert counts["merged_facts"] == 1 and counts["ended_facts"] == 0 and counts["past_versions"] == 1


def test_a_chain_is_rebound_to_the_root(provider):
    a = _call(provider, action="add", content="Alice Smith lives in Paris")["fact_id"]
    b = _call(provider, action="add", content="Alice Smith is living in Paris zorblat")["fact_id"]
    c = _call(provider, action="add", content="Alice Smith resides in Paris quxxle")["fact_id"]
    _call(provider, action="merge", fact_id=a, **{"from": [b]})
    result = _call(provider, action="merge", fact_id=c, **{"from": [a]})
    assert result["merged"] == [a]
    store = provider._store
    assert {tuple(r) for r in store.read("SELECT from_fact_id, into_fact_id FROM fact_merges")} == {(a, c), (b, c)}
    assert _call(provider, action="history", fact_id=b)["merged_into"] == c
    assert [m["fact_id"] for m in _call(provider, action="history", fact_id=c)["merged_from"]] == [a, b]
    assert _call(provider, action="add", content="Alice Smith is living in Paris zorblat")["fact_id"] == c


def test_export_lists_what_was_merged_into_the_facts_shown(provider):
    keep, copy = _pair(provider)
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    result = _call(provider, action="export")
    assert result["count"] == 1 and result["merged"] == [{"fact_id": copy, "into": keep, "merged_at": "2026-10-12 12:00:00"}]
    assert f"#{copy} → #{keep} (2026-10-12): Alice Smith is living in Paris by the Louvre zorblat" in result["markdown"]


# --- refusals: nothing changes ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("args, said", [
    ({"from": []}, "needs 'from'"),
    ({"from": ["x"]}, "needs 'from'"),
    ({"from": [1, 1]}, "twice"),
    ({"from": [999]}, "does not exist"),
    ({"from": list(range(100, 121))}, "at most 20"),
])
def test_a_bad_from_is_refused_without_changes(provider, args, said):
    keep, copy = _pair(provider)
    before = _snapshot(provider._store)
    result = _call(provider, action="merge", fact_id=keep, **args)
    assert said in result["error"]
    assert _snapshot(provider._store) == before


def test_the_keep_in_from_a_copy_as_keep_and_an_ended_fact_are_refused_without_changes(provider):
    keep, copy = _pair(provider)
    third = _call(provider, action="add", content="Alice Smith resides in Paris at the Louvre quxxle")["fact_id"]
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    ended = _call(provider, action="add", content="Alice Smith works at the Louvre")["fact_id"]
    _call(provider, action="end", fact_id=ended)
    before = _snapshot(provider._store)
    assert "into itself" in _call(provider, action="merge", fact_id=keep, **{"from": [keep, third]})["error"]
    assert f"was merged into #{keep}" in _call(provider, action="merge", fact_id=copy, **{"from": [third]})["error"]
    assert "has ended" in _call(provider, action="merge", fact_id=keep, **{"from": [third, ended]})["error"]
    assert "has ended" in _call(provider, action="merge", fact_id=ended, **{"from": [third]})["error"]
    other = _call(provider, action="add", content="Alice Smith lives in Paris, France")["fact_id"]
    assert f"was merged into #{keep}" in _call(provider, action="merge", fact_id=other, **{"from": [copy]})["error"]
    _call(provider, action="forget", fact_id=other)
    assert _snapshot(provider._store) == before


def test_a_read_only_store_and_a_subagent_refuse(tmp_path, clock):
    main = _provider(tmp_path)
    keep, copy = _pair(main)
    sub = _provider(tmp_path, agent_context="subagent")
    result = _call(sub, action="merge", fact_id=keep, **{"from": [copy]})
    assert "read-only" in result["error"]
    sub.shutdown()
    main.shutdown()
    MemoryStore._shared.clear()
    reader = MemoryStore(str(tmp_path / "memory_store.db"), hrr_dim=DIM, migrate=False)
    with pytest.raises(ValueError, match="read-only"):
        reader.merge_facts(keep, [copy])
    assert reader.count_facts() == 2
    reader.close()


def test_a_scheduled_job_merges_only_what_it_stored_in_this_session(tmp_path, clock):
    main = _provider(tmp_path)
    keep, old = _pair(main)
    main.shutdown()
    cron = _provider(tmp_path, agent_context="cron", platform="cron", session_id="cron1")
    new = _call(cron, action="add", content="Alice Smith resides in Paris at the Louvre quxxle", origin="said")["fact_id"]
    before = _snapshot(cron._store)
    refused = _call(cron, action="merge", fact_id=keep, **{"from": [old, new]})
    assert "only facts it stored in this session" in refused["error"] and f"#{old}" in refused["error"]
    assert _snapshot(cron._store) == before
    result = _call(cron, action="merge", fact_id=keep, **{"from": [new]})
    assert result["merged"] == [new]
    cron.shutdown()
    off = _provider(tmp_path, agent_context="cron", config={"cron_writes": "false"}, session_id="cron2")
    assert "read-only" in _call(off, action="merge", fact_id=keep, **{"from": [old]})["error"]
    off.shutdown()


def test_expect_refuses_a_version_the_caller_did_not_see(provider):
    keep, copy = _pair(provider)
    seen = {str(keep): provider._store._current_version(keep)["version_id"],
            str(copy): provider._store._current_version(copy)["version_id"]}
    _call(provider, action="change", fact_id=copy, content="Alice Smith is living in Paris by the Louvre, she said")
    before = _snapshot(provider._store)
    result = _call(provider, action="merge", fact_id=keep, **{"from": [copy]}, expect=seen)
    assert "changed since" in result["error"] and "Nothing was changed" in result["error"]
    assert result["changed_since"] == {str(copy): provider._store._current_version(copy)["version_id"]}
    assert _snapshot(provider._store) == before
    seen[str(copy)] = provider._store._current_version(copy)["version_id"]
    assert _call(provider, action="merge", fact_id=keep, **{"from": [copy]}, expect=seen)["merged"] == [copy]
    again = _call(provider, action="merge", fact_id=keep, **{"from": [copy]}, expect=seen)  # the same again: ok
    assert again["merged"] == [] and "error" not in again
    with pytest.raises(ChangedSince):
        provider._store.merge_facts(keep, [copy], expect={copy: 1})


# --- forget ------------------------------------------------------------------------------------------------------------------


def test_forget_of_the_keep_forgets_the_copies_with_a_tombstone_each(tmp_path, clock):
    provider = _provider(tmp_path)
    db = tmp_path / "memory_store.db"
    keep, copy = _pair(provider)
    third = _call(provider, action="add", content="Alice Smith resides in Paris at the Louvre quxxle")["fact_id"]
    _call(provider, action="merge", fact_id=keep, **{"from": [copy, third]})
    assert b"zorblat" in _raw_bytes(db)
    result = _call(provider, action="forget", fact_id=keep)
    assert result["forgotten"] is True and result["versions"] == 3 and result["scrub"] == "done"
    assert result["merged_copies"] == [copy, third] and "tombstone each" in result["hint"]
    store = provider._store
    for fact_id in (keep, copy, third):
        assert store.read("SELECT COUNT(*) FROM forgotten WHERE fact_id = ?", (fact_id,))[0][0] == 1
        assert _call(provider, action="history", fact_id=fact_id)["forgotten"] is True
    for text in ("Alice Smith is living in Paris by the Louvre zorblat", "Alice Smith resides in Paris at the Louvre quxxle",
                 "Alice Smith lives in Paris near the Louvre"):
        assert store.forgotten_match(text) is not None
    assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    for table in ("facts", "fact_versions", "fact_entities", "fact_confirmations"):
        assert store.read(f"SELECT COUNT(*) FROM {table}")[0][0] == 0
    assert store.read("SELECT COUNT(*) FROM entities WHERE name = 'Alice Smith'")[0][0] == 0
    for word in (b"zorblat", b"quxxle", b"louvre"):
        assert word not in _raw_bytes(db)
    cron = _provider(tmp_path, agent_context="cron", platform="cron", session_id="cron1")
    assert _call(cron, action="add", content="Alice Smith resides in Paris at the Louvre quxxle")["status"] == "forgotten"
    cron.shutdown()
    provider.shutdown()


def test_forget_of_a_copy_erases_its_history_only(tmp_path, clock):
    provider = _provider(tmp_path)
    keep, copy = _pair(provider)
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    result = _call(provider, action="forget", fact_id=copy)
    assert result["forgotten"] is True and result["versions"] == 1 and result["merged_into"] == keep
    assert f"forget {{fact_id: {keep}}}" in result["hint"]
    store = provider._store
    assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    assert store.read("SELECT COUNT(*) FROM fact_versions WHERE fact_id = ?", (copy,))[0][0] == 0
    assert _call(provider, action="history", fact_id=keep)["current"] is True
    assert "merged_from" not in _call(provider, action="history", fact_id=keep)
    assert _found(provider, "Alice Louvre") == [keep]
    assert b"zorblat" not in _raw_bytes(tmp_path / "memory_store.db")
    assert _call(provider, action="add", content="Alice Smith is living in Paris by the Louvre zorblat")["status"] == "added"
    provider.shutdown()


def test_forget_of_the_keep_with_a_blocked_scrub_is_retried(tmp_path, clock):
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    assert store._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    for i in range(30):
        store.add_fact(f"Person{i} Surname{i} lives in city number {i}")
    keep = store.add_fact("Alice Smith lives in Paris")
    copy = store.add_fact("Alice Smith is living in Paris zorblat")
    store.merge_facts(keep, [copy])
    other = sqlite3.connect(str(db), isolation_level=None)
    cur = other.execute("SELECT fact_id, content FROM facts")
    cur.fetchone()
    try:
        result = store.forget_fact(keep)
    finally:
        cur.close()
        other.close()
    assert result["forgotten"] is True and result["scrub"] == "pending" and result["merged_copies"] == [copy]
    assert store.read("SELECT COUNT(*) FROM forgotten")[0][0] == 2
    again = store.forget_fact(keep)
    assert again["already"] is True and again["scrub"] == "done"
    assert b"zorblat" not in _raw_bytes(db)
    store.close()


# --- a failure leaves nothing written ------------------------------------------------------------------------------------


def test_a_failure_on_the_second_copy_rolls_the_whole_merge_back(tmp_path, clock, monkeypatch):
    provider = _provider(tmp_path)
    keep, copy = _pair(provider)
    third = _call(provider, action="add", content="Alice Smith resides in Paris at the Louvre quxxle")["fact_id"]
    _call(provider, action="search", query="quxxle")  # the retriever caches the texts
    store = provider._store
    before = _snapshot(store)
    original = MemoryStore._close_version
    third_version = store._current_version(third)["version_id"]

    def failing(self, version_id, status, **kw):  # the second copy: the first is closed and gone already
        if version_id == third_version and status == "merged":
            raise sqlite3.OperationalError("disk I/O error")
        return original(self, version_id, status, **kw)

    monkeypatch.setattr(MemoryStore, "_close_version", failing)
    result = _call(provider, action="merge", fact_id=keep, **{"from": [copy, third]})
    assert "error" in result
    monkeypatch.setattr(MemoryStore, "_close_version", original)
    assert not store._conn.in_transaction
    assert _snapshot(store) == before
    assert store.read("SELECT value FROM meta WHERE key = 'history_schema'")[0][0] == "1"
    assert sorted(_found(provider, "Alice Louvre")) == [keep, copy, third]
    assert store.read("SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH 'zorblat'")[0][0] == 1
    assert "zorblat" in provider.prefetch("Is Alice Smith living in Paris by the Louvre zorblat?")
    provider.shutdown()
    MemoryStore._shared.clear()
    reopened = _provider(tmp_path)
    assert reopened._store.count_facts() == 3 and _snapshot(reopened._store)["fact_versions"] == before["fact_versions"]
    assert reopened._store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    reopened.shutdown()


def test_a_failed_commit_leaves_nothing_and_the_next_merge_works(tmp_path, clock):
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    assert store._conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower() == "delete"
    for i in range(5):
        store.add_fact(f"Alice Smith lives in Paris, flat {i}")
    keep = store.add_fact("Alice Smith lives in Paris")
    copy = store.add_fact("Alice Smith is living in Paris zorblat")
    before = _snapshot(store)
    other = sqlite3.connect(str(db), isolation_level=None)
    cur = other.execute("SELECT fact_id, content FROM facts")
    cur.fetchone()
    store._conn.execute("PRAGMA busy_timeout = 200")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.merge_facts(keep, [copy])
        assert not store._conn.in_transaction
    finally:
        cur.close()
        other.close()
    assert _snapshot(store) == before
    assert sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM fact_merges").fetchone()[0] == 0
    assert store.merge_facts(keep, [copy])["merged"] == [copy]
    assert sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM fact_merges").fetchone()[0] == 1
    store.close()


# --- scopes and the schema mark --------------------------------------------------------------------------------------


def test_user_scope_merges_inside_the_persons_own_store(tmp_path, clock):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": DIM,
                                                 "scope": "user"})
    provider.initialize(session_id="s", hermes_home=str(tmp_path), platform="telegram", user_id="111")
    keep, copy = _pair(provider)
    provider.on_turn_start(2, "hi", author_id="222", author_name="Bob")
    bobs = _call(provider, action="add", content="Bob Brown lives in Rome")["fact_id"]
    assert "does not exist" in _call(provider, action="merge", fact_id=bobs, **{"from": [copy]})["error"]
    provider.on_turn_start(3, "hi", author_id="111", author_name="Alice")
    assert _call(provider, action="merge", fact_id=keep, **{"from": [copy]})["merged"] == [copy]
    alice = provider._store
    assert alice.read("SELECT value FROM meta WHERE key = 'history_schema'")[0][0] == "2"
    provider.on_turn_start(4, "hi", author_id="222", author_name="Bob")
    assert provider._store.read("SELECT value FROM meta WHERE key = 'history_schema'")[0][0] == "1"
    assert provider._store.count_facts() == 1 and provider._store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    provider.shutdown()


def test_the_schema_mark_rises_with_the_first_merge_not_on_open(tmp_path, clock):
    provider = _provider(tmp_path)
    store = provider._store
    assert store.read("SELECT value FROM meta WHERE key = 'history_schema'")[0][0] == "1"
    keep, copy = _pair(provider)
    provider.shutdown()
    MemoryStore._shared.clear()
    provider = _provider(tmp_path)
    assert provider._store.read("SELECT value FROM meta WHERE key = 'history_schema'")[0][0] == "1"
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    assert provider._store.read("SELECT value FROM meta WHERE key = 'history_schema'")[0][0] == "2"
    _call(provider, action="forget", fact_id=keep)  # the mark stays: it says what the file may hold
    provider.shutdown()
    MemoryStore._shared.clear()
    provider = _provider(tmp_path)
    assert provider._store.read("SELECT value FROM meta WHERE key = 'history_schema'")[0][0] == "2"
    provider.shutdown()


def test_reconciliation_erases_the_copies_of_a_keep_an_older_release_removed(tmp_path, clock):
    """A keep whose row a release without history deleted (facts only): the current reconciliation erases it with a
    tombstone, and 0.19.0 then erases its merged copies the same way — on the next writer open."""
    provider = _provider(tmp_path)
    keep, copy = _pair(provider)
    _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    store = provider._store
    store._write("DELETE FROM fact_entities WHERE fact_id = ?", (keep,))
    store._write("DELETE FROM facts WHERE fact_id = ?", (keep,))
    orphan = _call(provider, action="add", content="Bob Brown lives in Rome")["fact_id"]
    twin = _call(provider, action="add", content="Bob Brown is living in Rome quxxle")["fact_id"]
    _call(provider, action="merge", fact_id=orphan, **{"from": [twin]})
    store._write("DELETE FROM fact_versions WHERE fact_id = ?", (twin,))  # an older forget(copy) left the merge row
    provider.shutdown()
    MemoryStore._shared.clear()
    reopened = _provider(tmp_path)
    store = reopened._store
    assert {r[0] for r in store.read("SELECT fact_id FROM forgotten")} == {keep, copy}
    assert store.read("SELECT COUNT(*) FROM fact_versions WHERE fact_id IN (?, ?)", (keep, copy))[0][0] == 0
    assert [tuple(r) for r in store.read("SELECT from_fact_id, into_fact_id FROM fact_merges")] == []
    assert store.forgotten_match("Alice Smith is living in Paris by the Louvre zorblat") is not None
    assert b"zorblat" not in _raw_bytes(tmp_path / "memory_store.db")
    assert _call(reopened, action="history", fact_id=orphan)["current"] is True
    reopened.shutdown()


def test_merge_is_serialised_with_the_other_operations(provider):
    """Two threads: a merge and a search on the same provider do not interleave inside the store."""
    keep, copy = _pair(provider)
    errors = []

    def run():
        try:
            for _ in range(5):
                _call(provider, action="search", query="Alice Louvre")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.01)
    result = _call(provider, action="merge", fact_id=keep, **{"from": [copy]})
    thread.join()
    assert result["merged"] == [copy] and not errors
    assert isinstance(provider._retriever, FactRetriever)
    with pytest.raises(NotFound):
        provider._store.merge_facts(keep, [99999])
