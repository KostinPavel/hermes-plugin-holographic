# -*- coding: utf-8 -*-
"""П.9: сет household заморожен и самосогласован (рецепт майнтейнера PR #10,
2026-10-22: один дом ~200 фактов, en/ru/zh, dev/test, лейблы на каждый вопрос).

self_check внутри household.py — полная проверка (цели существуют, срезы и
счётчики, nooverlap без общих слов). Здесь — пиннинг заморозки: изменение
данных без пересогласования должно быть видно в тесте.
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("household", REPO / "tools" / "household.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_self_check_passes():
    _load().self_check()  # SystemExit = нарушен инвариант данных


def test_counts_are_frozen():
    m = _load()
    assert len(m.FACTS) == 199            # ~200 фактов (рецепт)
    assert len(m.QUERIES) == 70           # dev 40 + test 30
    assert len(m.CHITCHAT) == 45          # dev 30 + test 15
    assert m.LANGS == ("en", "ru", "zh")


def test_every_labelled_target_exists_and_noanswer_is_empty():
    m = _load()
    for qid, q in m.QUERIES.items():
        assert all(t in m.FACTS for t in q["targets"]), qid
        assert (q["slice"] == "noanswer") == (q["targets"] == []), qid


def test_translations_share_the_split():
    # сплит один на весь id — переводы вопроса всегда в одном сплите (рецепт)
    m = _load()
    for qid, q in m.QUERIES.items():
        assert set(q) >= {"split", "slice", "targets", "en", "ru", "zh"}, qid
    assert {q["split"] for q in m.QUERIES.values()} == {"dev", "test"}
