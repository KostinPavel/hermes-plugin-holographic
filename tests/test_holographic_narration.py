"""NousResearch/hermes-agent#117790: besides tool-call JSON (refused since 0.3.0), the store filled with the agent's
own account of its steps — "I just read the file and updated the config as requested." — which is no fact about
anyone. Such a report is now refused at write with a reason: a first-person action done for the one the agent works
for ("as requested", "as you asked", "per your request"). The two together, so that a person's own past ("I moved to
Bangkok", "I switched to Postgres last week") and a request of someone else ("as requested by the course") stay
facts; a report without those words is still stored (the tool's description asks not to store task progress)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider

REPORTS = [
    "I just read the file and updated the config as requested.",  # the issue's example
    "I've updated the config as you asked.",
    "As requested, I ran the tests and fixed two failures.",
    "I have created the file per your request.",
    "I checked the logs as you asked; nothing unusual.",
]
FACTS = [
    "I moved to Bangkok in 2025.",
    "I switched to Postgres last week.",
    "I updated my passport in March.",
    "I sent the contract to Ksenia as she asked.",
    "I fixed the bike as you suggested.",
    "Anton deployed the site as requested by Ksenia.",
    "I installed Python 3.12 as requested by the course.",
    "I read The Pragmatic Programmer last year and loved it.",
    "I created a Telegram channel for the Bang Saray chat.",
    "I changed jobs in 2024 and now work at HR24.",
    "As you know, I moved to Bangkok in 2025.",
    "I asked Ksenia to update the config as requested by the client.",
]


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 512})
    p.initialize(session_id="s")
    yield p
    p.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


@pytest.mark.parametrize("text", REPORTS)
def test_a_report_of_the_agents_own_steps_is_refused(provider, text):
    result = _call(provider, action="add", content=text, origin="inferred")
    assert "error" in result and "report of steps" in result["error"], result
    assert _call(provider, action="list")["facts"] == []


@pytest.mark.parametrize("text", FACTS)
def test_a_persons_own_past_is_stored(provider, text):
    assert _call(provider, action="add", content=text, origin="said")["status"] == "added"


def test_change_refuses_it_too(provider):
    fact_id = _call(provider, action="add", content="The config uses port 8080", origin="said")["fact_id"]
    result = _call(provider, action="change", fact_id=fact_id, content=REPORTS[0])
    assert "error" in result and "report of steps" in result["error"], result
