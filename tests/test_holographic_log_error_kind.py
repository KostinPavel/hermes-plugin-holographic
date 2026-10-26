"""A log line names a database failure by its kind, never by its text (Codex review of #29, 2026-10-26).

SQLite quotes a column's value in some messages ("Could not decode to UTF-8 column 'content' with text '...'"), so
an exception's text can carry a fact past the screen into the log. PR #30 made the rerank stage log error_kind; three
older lines still printed the exception itself: the deferred recall counts in the store, the provider's flush of
those counts, and the prompt block when the facts cannot be counted.
"""

from __future__ import annotations

import logging
import sqlite3

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider

SECRET = "SENTINEL-4f1a the fact's own words"


def _quoting_error() -> sqlite3.OperationalError:
    return sqlite3.OperationalError(f"Could not decode to UTF-8 column 'content' with text '{SECRET}'")


def _provider(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "log.db"), "hrr_dim": 1024})
    provider.initialize(session_id="log")
    return provider


class _FailingWrites:
    """The store's connection, whose batch update fails with a message quoting a fact."""

    def __init__(self, conn):
        self._conn = conn

    def executemany(self, *args, **kwargs):
        raise _quoting_error()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_deferred_recall_counts_log_the_kind_not_the_text(tmp_path, caplog):
    provider = _provider(tmp_path)
    store = provider._store
    fact_id = store.add_fact("Яна любит жасминовый чай.")
    store._conn = _FailingWrites(store._conn)
    with caplog.at_level(logging.DEBUG, logger="hermes_plugin_holographic"):
        assert store.add_retrieval_counts({fact_id: 1}) is False
    assert "OperationalError" in caplog.text, "the line must still say what failed"
    assert SECRET not in caplog.text
    store._conn = store._conn._conn
    provider.shutdown()


def test_the_flush_of_recall_counts_logs_the_kind_not_the_text(tmp_path, caplog, monkeypatch):
    provider = _provider(tmp_path)
    fact_id = provider._store.add_fact("Яна любит жасминовый чай.")
    for key in list(provider._stores):
        provider._pending.setdefault(key, __import__("collections").Counter())[fact_id] += 1

    def fail(_counts):
        raise _quoting_error()

    monkeypatch.setattr(provider._store, "add_retrieval_counts", fail)
    with caplog.at_level(logging.DEBUG, logger="hermes_plugin_holographic"):
        provider._flush_counts()
    assert "OperationalError" in caplog.text, "the flush must have hit the failure"
    assert SECRET not in caplog.text
    monkeypatch.undo()
    provider.shutdown()


def test_the_prompt_block_logs_the_kind_not_the_text(tmp_path, caplog, monkeypatch):
    provider = _provider(tmp_path)

    def fail():
        raise _quoting_error()

    monkeypatch.setattr(provider._store, "count_facts", fail)
    with caplog.at_level(logging.WARNING, logger="hermes_plugin_holographic"):
        block = provider.system_prompt_block()
    assert "could not be read" in block
    assert "OperationalError" in caplog.text
    assert SECRET not in caplog.text
    monkeypatch.undo()
    provider.shutdown()
