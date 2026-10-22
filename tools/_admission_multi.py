# -*- coding: utf-8 -*-
"""Этап 0 (message-admission): multilingual — держится ли z-пол и правила на 18 языках.

По каждому языку (своя база из 20 фактов, как в multi-сете): 20 real + 5 chit.
Считает те же статистики, что и _admission_probe: rule/worth_asking, voc-правило,
z, cmax, s1. Главные вопросы:
  1) где лежит минимальный z реальных запросов по языкам (глобальный порог 2.73
     безопасен, если min real z по ВСЕМ языкам > 2.73);
  2) сколько chit убивает глобальный порог по каждому языку;
  3) какие языки не покрыты правилами (worth_asking/voc).

    PYTHONPATH=~/.hermes/hermes-agent HERMES_AGENT_REPO=~/.hermes/hermes-agent \
        python -I tools/_admission_multi.py
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

G_GLOBAL = 2.73  # глобальный порог z из _admission_probe (бюджет 0 на ru/en/zh)


def voc_admit(msg: str) -> bool:
    if hemb.worth_asking(msg, before_turn=True):
        return True
    clauses = [c for c in htext._CLAUSE.split(msg or "") if c.strip()]
    if len(clauses) > 1 and htext.is_greeting(" ".join(clauses[1:])):
        return False
    head, sep, rest = (msg or "").partition(",")
    return not (sep and len(head.split()) <= 3 and htext.is_greeting(rest.strip()))


def collect_lang(lang: str, data) -> list[dict]:
    db = Path(tempfile.gettempdir()) / f"adm_ml_{lang}.db"
    for s in ("", "-shm", "-wal"):
        try:
            Path(str(db) + s).unlink()
        except OSError:
            pass
    p = pp.make_provider(db, "gate_only", "on")
    for _key, (text, entities) in data.CORPUS[lang].items():
        p.handle_tool_call("fact_store", {"action": "add", "content": text, "entities": entities})
    assert pp.catch_up_vectors(p), f"векторы не готовы ({lang})"
    width = max(int(getattr(hmod, "_PREFETCH_CANDIDATES", 15) * getattr(hmod, "_RERANK_WIDEN", 4.0)),
                int(p._rescorer.settings.top_k))
    rows: list[dict] = []
    for i, (query, relevant) in enumerate(data.QUERIES[lang]):
        p.on_session_switch(f"m{i}", reset=True)
        meaning = p._meaning_of(query, before_turn=True)
        cmax = z = None
        if meaning and meaning.sims:
            best = max(meaning.sims, key=meaning.sims.get)
            cmax = float(meaning.sims[best])
            z = float(meaning.z(best))
        s1 = None
        cands = p._retriever.search(query, min_trust=p._min_trust, limit=width, meaning=meaning)
        if cands:
            ranked = p._rescorer.scores(query, [c.get("content") or "" for c in cands])
            if ranked:
                s1 = float(ranked[0][1])
        rows.append({"lang": lang, "kind": "real" if relevant else "chit", "q": query,
                     "rule": hemb.worth_asking(query, before_turn=True),
                     "voc": voc_admit(query), "cmax": cmax, "z": z, "s1": s1})
    p.shutdown()
    return rows


def main() -> None:
    data = pp.load_multilingual()
    rows: list[dict] = []
    for lang in data.CORPUS:
        rows.extend(collect_lang(lang, data))
        real = [r for r in rows if r["lang"] == lang and r["kind"] == "real"]
        chit = [r for r in rows if r["lang"] == lang and r["kind"] == "chit"]
        zr = [r["z"] for r in real if r["z"] is not None]
        zc = [r["z"] for r in chit if r["z"] is not None]
        min_real = min(zr) if zr else float("nan")
        killed = sum(1 for r in chit if r["z"] is None or r["z"] < G_GLOBAL)
        lost = sum(1 for r in real if r["z"] is None or r["z"] < G_GLOBAL)
        rule_killed = sum(1 for r in chit if not r["rule"])
        print(f"{lang}: min_real_z={min_real:6.3f}  chit@z2.73={killed}/5  lost={lost}/20  "
              f"rule_kill_chit={rule_killed}/5  chit_z_med={sorted(zc)[len(zc)//2] if zc else float('nan'):6.3f}",
              flush=True)
    # jsonl-артефакт (не перезаписывать будущие анализы без GPU)
    out = _REPO / "results" / "admission_multi.jsonl"
    import json as _json
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(_json.dumps(r, ensure_ascii=False) + "\n")
    print(f"артефакт: {out} ({len(rows)} строк)")

    real = [r for r in rows if r["kind"] == "real"]
    chit = [r for r in rows if r["kind"] == "chit"]
    # свип глобального порога: потери real (глобально и максимум по одному языку) vs убито chit
    print("\n-- свип глобального порога z --")
    print("| t | lost real | max lost/язык | killed chit |")
    print("|---|---|---|---|")
    for t in (2.2, 2.3, 2.35, 2.4, 2.5, 2.6, 2.7, 2.73):
        lost = [r for r in real if r["z"] is not None and r["z"] < t]
        killed = [r for r in chit if r["z"] is None or r["z"] < t]
        per = {}
        for r in lost:
            per[r["lang"]] = per.get(r["lang"], 0) + 1
        print(f"| {t} | {len(lost)}/360 | {max(per.values()) if per else 0} | {len(killed)}/90 |")

    # общий итог по глобальному порогу
    lost = [r for r in real if r["z"] is None or r["z"] < G_GLOBAL]
    killed = [r for r in chit if r["z"] is None or r["z"] < G_GLOBAL]
    print(f"\nИТОГО 18 языков: real {len(real)}, chit {len(chit)}")
    print(f"глобальный z-пол {G_GLOBAL}: потеря real {len(lost)}/{len(real)}, убито chit {len(killed)}/{len(chit)}")
    if lost:
        for r in lost:
            print(f"  LOSS [{r['lang']}] z={r['z']} :: {r['q'][:70]}")
    # языки, где rule/voc не убивают НИ ОДНОГО chit (словарь не покрыт)
    weak = {}
    for r in chit:
        if r["rule"] and r["voc"]:
            weak.setdefault(r["lang"], []).append(r["q"])
    if weak:
        print("\nchit, который правила НЕ ловят (правила → True):")
        for lang, qs in sorted(weak.items()):
            for q in qs:
                print(f"  [{lang}] {q[:70]}")


if __name__ == "__main__":
    main()