"""0.14.3: a column another program left in the facts table never reaches a tool result (0.14.2: binary ones only);
a 400 to `dimensions` is named (0.14.2).

Both reported on a live store that moved to 0.14.1 from a hand-rolled embedding deployment (issue #3)."""
from __future__ import annotations

import json
import sqlite3
import urllib.error

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings, retrieval

FACTS = ["The garden shed is painted green and has a red door",
         "The backup of the garden server runs every night at three",
         "The garden hose is kept behind the shed"]


def _provider(tmp_path, **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64, **config})
    p.initialize(session_id="s1")
    return p


def _legacy_column(tmp_path):
    """What the earlier deployment left behind: a BLOB column of its own vectors (NULL on one fact) and a text column."""
    con = sqlite3.connect(tmp_path / "memory_store.db")
    con.execute("ALTER TABLE facts ADD COLUMN embedding BLOB")
    con.execute("ALTER TABLE facts ADD COLUMN legacy_note TEXT")
    con.execute("UPDATE facts SET embedding = randomblob(64), legacy_note = 'private metadata'")
    con.execute("UPDATE facts SET embedding = NULL WHERE fact_id = (SELECT MIN(fact_id) FROM facts)")
    con.commit()
    con.close()


def test_a_foreign_blob_column_does_not_break_search(tmp_path):
    p = _provider(tmp_path)
    ids = [p._store.add_fact(f) for f in FACTS]
    _legacy_column(tmp_path)
    for query in ("garden shed", "when does the garden backup run?", "hose"):
        raw = p.handle_tool_call("fact_store", {"action": "search", "query": query})
        found = json.loads(raw)  # 0.14.1: "Object of type bytes is not JSON serializable" on every search
        assert found["results"], query
        assert all(not {"embedding", "legacy_note", "hrr_vector"} & set(f) for f in found["results"])
        assert "private metadata" not in raw
    rows = p._retriever._rows(ids)
    assert len(rows) == 3 and all(not {"embedding", "legacy_note"} & set(r) for r in rows)
    assert f"#{ids[2]} " in p.prefetch("where is the garden hose kept?", session_id="x")
    p.shutdown()


def test_our_own_vector_still_scores(tmp_path):
    """The filter keeps hrr_vector: the holographic term still reads it (and it is popped before output)."""
    p = _provider(tmp_path)
    for f in FACTS:
        p._store.add_fact(f)
    _legacy_column(tmp_path)
    cands = p._retriever._fts_candidates("garden shed", None, 0.0, 10)
    assert cands and all(isinstance(c["hrr_vector"], bytes) and c["hrr_vector"] for c in cands)
    assert all(not {"embedding", "legacy_note"} & set(c) for c in cands)
    p.shutdown()


def test_the_allowlist_covers_every_column_of_ours(tmp_path):
    """A column the plugin itself adds to facts (or to the as_of projection) must be in _ROW_KEYS, or reads lose it."""
    p = _provider(tmp_path)
    p._store.add_fact(FACTS[0])
    keys = retrieval._ROW_KEYS
    facts_cols = {r[1] for r in p._store.read("PRAGMA table_info(facts)")}
    assert facts_cols <= keys, facts_cols - keys
    cur = sqlite3.connect(tmp_path / "memory_store.db").execute(
        f"SELECT * FROM {retrieval._AS_OF_SOURCE} LIMIT 0", (0.5, "2026-01-01", "2026-01-01"))
    as_of_cols = {d[0] for d in cur.description}
    assert as_of_cols <= keys, as_of_cols - keys
    p.shutdown()


class _Refuses400:
    def open(self, request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)


ON = {"embeddings": "on", "embedding_url": "https://embeddings.example.net/v1", "embedding_model": "bge-m3",
      "embedding_key_env": "TEST_EMBEDDINGS_KEY"}


@pytest.mark.parametrize("dimensions, hinted", [(1024, True), (None, False)])
def test_a_400_to_dimensions_names_the_setting(monkeypatch, dimensions, hinted):
    monkeypatch.setattr(embeddings, "_OPENER", _Refuses400())
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    config = {**ON, **({"embedding_dimensions": dimensions} if dimensions else {})}
    e = embeddings.Embedder(embeddings.Settings(config))
    assert e.embed(["a"]) is None
    status = e.status()
    assert status.startswith("paused after an error (HTTP 400")
    assert ("embedding_dimensions is set" in status) is hinted
    assert "sk-test" not in status
