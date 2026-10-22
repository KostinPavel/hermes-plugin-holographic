# -*- coding: utf-8 -*-
"""Этап 0 (message-admission): какое измерение разделяет реальные запросы и чичат.

Кандидаты (все — статистика СООБЩЕНИЯ, порог убивает сообщение целиком):
  rule    worth_asking(query, before_turn=True)         — правила сейчас (длина+формулы)
  voc     правило с вокативом: убрать первую clause и перепроверить is_greeting
  cmax    max cosine (сообщение, база)                  — семантика к базе
  z       z-score лучшего косинуса                      — семантика (нормировка)
  s1      top-1 скор реранка по пулу                    — реранкер
  gap     s1 - s2                                       — острота пика

Свип: для бюджета потерь реальных запросов (0/1/2/3/5) — сколько чичата убито.

    PYTHONPATH=~/.hermes/hermes-agent HERMES_AGENT_REPO=~/.hermes/hermes-agent \
        python -I tools/_admission_probe.py
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_REPO / "tools"))

import provenance_probe as pp  # noqa: E402

hmod = importlib.import_module("hermes_plugin_holographic")
htext = importlib.import_module("hermes_plugin_holographic.text")
hemb = importlib.import_module("hermes_plugin_holographic.embeddings")


def voc_admit(msg: str) -> bool:
    """Правило с вокативом: «Имя, <формула>» — формула без вокатива."""
    if hemb.worth_asking(msg, before_turn=True):
        return True
    clauses = [c for c in htext._CLAUSE.split(msg or "") if c.strip()]
    if len(clauses) > 1 and htext.is_greeting(" ".join(clauses[1:])):
        return False
    head, sep, rest = (msg or "").partition(",")
    return not (sep and len(head.split()) <= 3 and htext.is_greeting(rest.strip()))


def collect(set_name: str) -> list[dict]:
    contrast = pp.load_contrast(set_name)
    db = Path(tempfile.gettempdir()) / f"adm_{set_name}.db"
    for s in ("", "-shm", "-wal"):
        try:
            Path(str(db) + s).unlink()
        except OSError:
            pass
    p = pp.make_provider(db, "gate_only", "on")
    for _key, (text, entities) in contrast.FACTS.items():
        p.handle_tool_call("fact_store", {"action": "add", "content": text, "entities": entities})
    assert pp.catch_up_vectors(p), "векторы не готовы"
    rows: list[dict] = []
    msgs = [(q, "real", "real") for pair in contrast.PAIRS for (q, _t) in pair["queries"]]
    msgs += [(q, "chit", cat) for q, cat in contrast.CHITCHAT]
    width = max(int(getattr(hmod, "_PREFETCH_CANDIDATES", 15) * getattr(hmod, "_RERANK_WIDEN", 4.0)),
                int(p._rescorer.settings.top_k))
    for i, (msg, kind, cat) in enumerate(msgs):
        p.on_session_switch(f"a{i}", reset=True)
        meaning = p._meaning_of(msg, before_turn=True)
        cmax = z = None
        if meaning and meaning.sims:
            best = max(meaning.sims, key=meaning.sims.get)
            cmax = float(meaning.sims[best])
            z = float(meaning.z(best))
        s1 = s2 = None
        cands = p._retriever.search(msg, min_trust=p._min_trust, limit=width, meaning=meaning)
        if cands:
            ranked = p._rescorer.scores(msg, [c.get("content") or "" for c in cands])
            if ranked and len(ranked) >= 1:
                s1 = float(ranked[0][1])
                if len(ranked) >= 2:
                    s2 = float(ranked[1][1])
        rows.append({"set": set_name, "kind": kind, "cat": cat, "msg": msg,
                     "rule": hemb.worth_asking(msg, before_turn=True),
                     "voc": voc_admit(msg),
                     "cmax": cmax, "z": z, "s1": s1,
                     "gap": None if s1 is None or s2 is None else s1 - s2})
    p.shutdown()
    return rows


def sweep(vals_real: list, vals_chit: list) -> dict[int, tuple[int, str]]:
    """Лучший (убито чичата) при бюджете потерь реальных; kill = None-сигнал или v < t."""
    pts = sorted({v for v in vals_real + vals_chit if v is not None})
    grid = [float("-inf")] + [(pts[i] + pts[i + 1]) / 2 for i in range(len(pts) - 1)] + [pts[-1] + 1e-6]
    best: dict[int, tuple[int, str]] = {}
    for t in grid:
        lost = sum(1 for v in vals_real if v is None or v < t)
        killed = sum(1 for v in vals_chit if v is None or v < t)
        for b in (0, 1, 2, 3, 5):
            if lost <= b and (b not in best or killed > best[b][0]):
                best[b] = (killed, str(t))
    return best


def main() -> None:
    rows: list[dict] = []
    for set_name in ("contrast_ru", "contrast_en", "contrast_zh"):
        rows.extend(collect(set_name))
    real = [r for r in rows if r["kind"] == "real"]
    chit = [r for r in rows if r["kind"] == "chit"]
    print(f"сообщений: real {len(real)}, chit {len(chit)}")
    print("\n-- медианы / None по классам --")
    for stat in ("cmax", "z", "s1", "gap"):
        rv = [r[stat] for r in real if r[stat] is not None]
        cv = [r[stat] for r in chit if r[stat] is not None]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")  # noqa: E731
        print(f"{stat:>5}: real {med(rv):.3f} (n={len(rv)})  chit {med(cv):.3f} (n={len(cv)})")
    for stat in ("rule", "voc"):
        lost = sum(1 for r in real if not r[stat])
        killed = sum(1 for r in chit if not r[stat])
        print(f"{stat:>5}: убивает chit {killed}/{len(chit)}, теряет real {lost}/{len(real)}")
    print("\n-- свип порога: лучший шум-килл при бюджете потерь real --")
    print("| стат | б0 | б1 | б2 | б3 | б5 |")
    print("|---|---|---|---|---|---|")
    for stat in ("cmax", "z", "s1", "gap"):
        best = sweep([r[stat] for r in real], [r[stat] for r in chit])
        cells = " | ".join(f"{best.get(b, (0,))[0]} (t={best.get(b, (0, ''))[1]})" for b in (0, 1, 2, 3, 5))
        print(f"| {stat} | {cells} |")
    # объединённый этап 0: правила (rule ∪ voc) И z-пол — лучший при бюджете
    def kill(r: dict, t: float) -> bool:
        return not r["voc"] or r["z"] is None or r["z"] < t

    pts = sorted({r["z"] for r in rows if r["z"] is not None})
    grid = [float("-inf")] + [(pts[i] + pts[i + 1]) / 2 for i in range(len(pts) - 1)] + [pts[-1] + 1e-6]
    print("\n-- объединение: (rule ∪ voc) И z-пол --")
    for t in grid:
        lost = sum(1 for r in real if kill(r, t))
        if lost <= 2:
            killed = sum(1 for r in chit if kill(r, t))
            print(f"  t={t:.3f}: убито chit {killed}/{len(chit)}, потеря real {lost}/{len(real)}")
    # разбивка выживших по категориям при лучшем t (бюджет 0)
    t0 = max((t for t in grid if sum(1 for r in real if kill(r, t)) == 0), default=0.0)
    surv: dict[str, list[int]] = {}
    for r in chit:
        cat = r["cat"]
        slot = surv.setdefault(cat, [0, 0])
        slot[1] += 1
        if not kill(r, t0):
            slot[0] += 1
    print(f"\n-- выжившие чичат по категориям при t={t0:.3f} (бюджет 0) --")
    for cat in sorted(surv):
        live, total = surv[cat]
        print(f"  {cat}: выжило {live}/{total}")
    print("\n-- примеры убитых при t0 (по 4 на категорию) --")
    shown: dict[str, int] = {}
    for r in chit:
        if kill(r, t0) and shown.get(r["cat"], 0) < 4:
            shown[r["cat"]] = shown.get(r["cat"], 0) + 1
            print(f"  [{r['cat']}] z={r['z']} s1={r['s1']} :: {r['msg'][:80]}")


if __name__ == "__main__":
    main()