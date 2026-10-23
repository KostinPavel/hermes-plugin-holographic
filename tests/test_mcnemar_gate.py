"""Fixture for tools/mcnemar_gate.py (п.2 фикс-листа 14:01): partial per-target loss and unpaired rows.

A multi-target question that keeps one target and drops the other still counts as a hit (c == 0) — branch B
must refuse it on per-target losses against off in the top five. Rows not paired with off must be printed,
not silently skipped (the pairing could show zero losses while incomplete)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "mcnemar_gate.py"
_spec = importlib.util.spec_from_file_location("mcnemar_gate", _TOOL)
assert _spec and _spec.loader
mcnemar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcnemar)


def _row(mode, qid, keys, targets, *, hit5=True, confused=False):
    return {"set": "household", "lang": "en", "qid": qid, "mode": mode, "kind": "real",
            "hit5": hit5, "keys": keys, "targets": targets, "confused": confused,
            "out_n": len(keys)}


def _chit(mode, qid):
    return {"lang": "en", "qid": qid, "mode": mode, "kind": "chit", "keys": [], "targets": [],
            "hit5": None, "confused": False, "out_n": 0}


def test_partial_target_loss_and_unpaired_rows(tmp_path, capsys):
    rows = [
        # q1: два таргета; gate_only держит hit5 (t1 в top-5), но t2 вытолкнут на ШЕСТУЮ позицию —
        # вопрос-level c == 0, а per-target потеря есть.
        _row("off", "q1", ["t1", "t2", "f1", "f2", "f3"], ["t1", "t2"], confused=True),
        _row("gate_only", "q1", ["t1", "f1", "f2", "f3", "f4", "t2"], ["t1", "t2"]),
        _row("gate_min2", "q1", ["t1", "t2", "f1", "f2", "f3"], ["t1", "t2"]),
        # q2: есть только у gate_only — unpaired-строка, её нельзя молча пропускать.
        _row("gate_only", "q2", ["t5"], ["t5"]),
        _chit("off", "c1"), _chit("gate_only", "c2"), _chit("gate_min2", "c3"),
    ]
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    mcnemar.report(str(path))
    out = capsys.readouterr().out

    gate_line = next(line for line in out.splitlines() if line.startswith("gate_only"))
    min2_line = next(line for line in out.splitlines() if line.startswith("gate_min2"))
    # поправка 14:01: вопрос не потерян (c == 0), но таргет потерян — ветки B нет
    assert gate_line.split()[-1] == "-", gate_line
    # контроль, что фикстура вообще может дать B (иначе тест доказывал бы только подавление)
    assert min2_line.split()[-1] == "B", min2_line
    # непарная строка печатается явным счётчиком, а не проглатывается
    assert "UNPAIRED" in out
    assert "1" in out.split("UNPAIRED", 1)[1]


def test_unpaired_rows_fail_closed(tmp_path, capsys):
    """Полная парность — иначе B не ставится, даже когда потерь не видно."""
    rows = [
        # все прочие условия ветки B выполнены: потерь нет (l5 = 0), confused уменьшен, neg равны —
        # остаётся единственная причина не ставить B: неполная парность (UNPAIRED).
        _row("off", "q1", ["t1"], ["t1"], confused=True),
        _row("gate_only", "q1", ["t1"], ["t1"]),
        _row("gate_only", "q2", ["t2"], ["t2"]),  # только у gate_only
        _chit("off", "c1"), _chit("gate_only", "c2"),
    ]
    path = tmp_path / "unpaired.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    mcnemar.report(str(path))
    out = capsys.readouterr().out
    gate_line = next(line for line in out.splitlines() if line.startswith("gate_only"))
    assert gate_line.split()[-1] == "-", gate_line
    assert "UNPAIRED" in out