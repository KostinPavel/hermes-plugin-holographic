"""Issue #23 (Praefectus35): the keyword limits were silent. At most 12 distinct keywords of up to 40 characters are
kept; a longer one was dropped and the 13th on were cut, and neither add nor update said so. The result now lists the
keywords kept and, when some were not, their positions in the list given and why — never their text."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider

LONG = "a keyword phrase much longer than forty characters in all"  # 57 characters
FIFTEEN = ["birth date", "anniversary", LONG, "день рождения", "cumpleaños", "party", "cake", "May", "spring",
           "family", "mother", "Birth Date", "gift", "celebration", "flowers"]  # #12 repeats #1; #15 is the 13th distinct


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 512})
    p.initialize(session_id="s")
    yield p
    p.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def test_add_reports_what_was_not_kept(provider):
    """The case of the issue: more than 12, one of them too long."""
    result = _call(provider, action="add", content="Mom's birthday is May 3", origin="said", keywords=FIFTEEN)
    assert result["status"] == "added"
    kept = [k for i, k in enumerate(FIFTEEN, 1) if i not in (3, 12, 15)]
    assert result["keywords"] == kept
    dropped = result["keywords_dropped"]
    assert "#3" in dropped and "longer than 40 characters" in dropped
    assert "#15" in dropped and "limit of 12" in dropped
    assert "#12" not in dropped  # a repeat is kept once, nothing is lost
    assert LONG not in json.dumps(result, ensure_ascii=False)  # positions, not text


def test_add_within_the_limits_reports_the_list_and_nothing_dropped(provider):
    result = _call(provider, action="add", content="The car is serviced every April", origin="said",
                   keywords=["maintenance", "техобслуживание", "garage"])
    assert result["keywords"] == ["maintenance", "техобслуживание", "garage"]
    assert "keywords_dropped" not in result


def test_without_keywords_nothing_is_reported(provider):
    result = _call(provider, action="add", content="Lena likes green tea", origin="said")
    assert "keywords" not in result and "keywords_dropped" not in result


def test_update_reports_the_list_that_replaced_the_old_one(provider):
    fact_id = _call(provider, action="add", content="Mom's birthday is May 3", origin="said")["fact_id"]
    result = _call(provider, action="update", fact_id=fact_id, keywords=FIFTEEN)
    assert result["updated"]
    assert len(result["keywords"]) == 12
    assert "#3" in result["keywords_dropped"] and "#15" in result["keywords_dropped"]


def test_a_comma_separated_string_is_counted_the_same_way(provider):
    fact_id = _call(provider, action="add", content="Mom's birthday is May 3", origin="said")["fact_id"]
    result = _call(provider, action="update", fact_id=fact_id, keywords=", ".join(FIFTEEN))
    assert len(result["keywords"]) == 12
    assert "#3" in result["keywords_dropped"]


def test_change_reports_the_keywords_of_the_new_text(provider):
    fact_id = _call(provider, action="add", content="Lena lives in Paris", origin="said")["fact_id"]
    result = _call(provider, action="change", fact_id=fact_id, content="Lena lives in Lyon", origin="said",
                   keywords=["home", "city", LONG])
    assert result["keywords"] == ["home", "city"]
    assert "#3" in result["keywords_dropped"]


def test_add_of_an_existing_fact_says_its_keywords_were_not_set(provider):
    """The same text again returns the existing fact and leaves its keywords as they were: say so, and how to set
    them, instead of reporting a list that was never written."""
    _call(provider, action="add", content="Lena likes green tea", origin="said")
    result = _call(provider, action="add", content="Lena likes green tea", origin="said", keywords=["drink"])
    assert result["status"] == "exists"
    assert "keywords" not in result
    assert "update" in result["keywords_not_set"]


def _stored(provider, fact_id):
    """The keywords the store holds for a fact, as recall reads them."""
    from hermes_plugin_holographic import _split_tags
    return _split_tags(provider._store.fact_tags(fact_id))[1]


def test_the_report_is_what_the_store_holds(provider):
    result = _call(provider, action="add", content="Mom's birthday is May 3", origin="said", keywords=FIFTEEN)
    assert result["keywords"] == _stored(provider, result["fact_id"])


def test_a_repeat_of_a_keyword_past_the_limit_is_not_kept_either(provider):
    """Codex review of PR #27: a repeat of a word that did not fit was read as "kept once" and left out of the
    report, though neither copy was kept."""
    twelve = [f"word{i}" for i in range(1, 13)]
    result = _call(provider, action="add", content="Mom's birthday is May 3", origin="said",
                   keywords=twelve + ["extra", "EXTRA"])
    assert result["keywords"] == twelve
    assert result["keywords_dropped"] == "2 not kept: #13, #14 past the limit of 12"


def test_change_onto_another_facts_text_does_not_report_keywords_it_did_not_write(provider):
    """Codex review of PR #27: changing a fact into the text another one already holds returns that other fact and
    leaves its keywords alone; the result must not claim the new list."""
    a = _call(provider, action="add", content="Lena lives in Paris", origin="said")["fact_id"]
    b = _call(provider, action="add", content="Lena lives in Lyon", origin="said", keywords=["old"])["fact_id"]
    result = _call(provider, action="change", fact_id=a, content="Lena lives in Lyon", origin="said",
                   keywords=["new"])
    assert result["status"] == "exists" and result["fact_id"] == b
    assert "keywords" not in result
    assert f"#{b}" in result["keywords_not_set"]
    assert _stored(provider, b) == ["old"]


def test_update_with_an_empty_list_reports_the_keywords_cleared(provider):
    fact_id = _call(provider, action="add", content="Lena likes green tea", origin="said", keywords=["drink"])["fact_id"]
    result = _call(provider, action="update", fact_id=fact_id, keywords=[])
    assert result["keywords"] == [] and "keywords_dropped" not in result
    assert _stored(provider, fact_id) == []
