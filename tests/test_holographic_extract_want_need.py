"""Regression: "I want/need" over-matched the user_pref pattern.

The pattern `\\bI\\s+(?:prefer|like|love|use|want|need)\\s+\\S` treats a
one-off intent as a durable preference. The real-world failure (observed in a
live store, 2026-10): a user who pastes a model's English draft to have it
translated sends that draft as a role="user" message, and the MODEL's
reasoning sentences — "Plus, I want to ask you about details like the
intended date and format", "So I need to find the 2023 production", "Wait,
but I need the 2023 production for B" — were harvested as user_pref facts.
They state no preference, carry no keywords, and pollute recall.

Fix: drop want/need from the English preference pattern. prefer/like/love/use
still carry the durable signal ("I prefer X over Y", "I use Z daily");
"want"/"need" almost always state a transient intent, in the user's own words
as much as in a pasted model draft.
"""

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider


def _make_provider(tmp_path, **config):
    base = {"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    base.update(config)
    provider = HolographicMemoryProvider(config=base)
    provider.initialize(session_id="test-session")
    return provider


def _user(content):
    return {"role": "user", "content": content}


# The exact sentences that landed as user_pref facts in a live store.
PASTED_MODEL_DRAFTS = [
    "Plus, I want to ask you about details like the intended date and format (in person in Korea, or online).",
    "So I need to find the 2023 production.",
    "Wait, but I need the 2023 production for B.",
]


@pytest.mark.parametrize("message", PASTED_MODEL_DRAFTS)
def test_transient_want_need_intent_is_not_stored(tmp_path, message):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(message)])
    assert [f["content"] for f in provider._store.list_facts(limit=100)] == []
    provider.shutdown()


def test_a_pasted_draft_around_a_real_preference_keeps_only_the_preference(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(
        "Translate this for me: Plus, I want to ask you about the venue. "
        "I prefer short answers with sources."
    )])
    facts = [f["content"] for f in provider._store.list_facts(limit=100)]
    assert facts == ["I prefer short answers with sources."]
    provider.shutdown()


def test_durable_use_preference_is_still_extracted(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user("I use Dark mode in every editor on my machines.")])
    facts = [f["content"] for f in provider._store.list_facts(limit=100)]
    assert facts == ["I use Dark mode in every editor on my machines."]
    provider.shutdown()
