# -*- coding: utf-8 -*-
"""Почему meaning-поток молчит: трассировка _meaning_of/by_meaning по реальным запросам
(en против ru). Ничего не патчит — только наблюдает.

Запуск:
    PYTHONPATH=~/.hermes/hermes-agent HERMES_AGENT_REPO=~/.hermes/hermes-agent \
        python -I tools/_en_meaning_debug.py [--set contrast_en --cmp contrast_ru]
"""
from __future__ import annotations

import argparse
import json
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

import provenance_probe as pp  # noqa: E402  (сам setup репо + load_contrast/make_provider/catch_up)
from hermes_plugin_holographic import embeddings  # noqa: E402


def fill_store(p, contrast):
    back = {}
    for key, (text, entities) in contrast.FACTS.items():
        r = json.loads(p.handle_tool_call(
            "fact_store", {"action": "add", "content": text, "entities": entities}))
        back[int(r["fact_id"])] = key
    return back


def _relevant(p, query):
    """Лексическая адмиссия, как её делает пайплайн (после safety-фильтра)."""
    cand = p._retriever.search(query, min_trust=p._min_trust, limit=p._rerank_width() if hasattr(p, "_rerank_width") else 40)
    return p._retriever.relevant(query, cand, p._min_coverage)


def trace(p, back, query, target_key):
    """Одна причина отказа meaning-потока на запросе."""
    out = {"q": query, "target": target_key}
    if not embeddings.worth_asking(query, before_turn=True):
        out["reason"] = "worth_asking=False"
        return out
    if not p._embedder.usable():
        out["reason"] = "embedder unusable"
        return out
    idx = p._retriever.vector_index(p._embedding.tag)
    if not idx.ready():
        out["reason"] = "index not ready"
        return out
    try:
        vector = p._embedder.query(query)
    except Exception as exc:
        out["reason"] = f"query() raised: {exc}"
        return out
    if vector is None:
        out["reason"] = "vector=None"
        return out
    meaning = p._retriever.meaning(vector, p._embedding.tag, p._embedding.weight)
    if meaning is None:
        out["reason"] = "retriever.meaning=None"
        return out
    n = len(meaning.sims)
    n_adm_eligible = max(n, 2)
    zbar = embeddings.z_bar(n_adm_eligible, embeddings.RECALL_MARGIN) if n >= embeddings.MIN_VECTORS_FOR_Z else None
    out["vectors"] = n
    out["z_bar"] = None if zbar is None else round(zbar, 3)
    # топ-5 по симметрии: sim / z / standout
    top = meaning.ranked()[:5]
    out["top"] = [dict(key=back.get(fid, "?"), sim=round(meaning.sims[fid], 4),
                       z=round(meaning.z(fid), 3),
                       standout=meaning.standout(fid, embeddings.RECALL_MARGIN,
                                                 p._embedding.min_score))
                  for fid, _sim in top]
    admits = p._retriever.by_meaning(
        meaning, exclude=set(), min_trust=p._min_trust,
        limit=p._embedding.recall_facts + 3, margin=embeddings.RECALL_MARGIN,
        min_score=p._embedding.min_score)
    out["by_meaning"] = [back.get(f["fact_id"], "?") for f in admits]
    # pipeline-семантика (_unified_stream): exclude = лексические адмиссы (coverage)
    lexical = p._retriever.relevant(query, [], min_trust=p._min_trust, limit=1,
                                    meaning=None) if False else _relevant(p, query)
    have = {f["fact_id"] for f in lexical}
    admits_h = p._retriever.by_meaning(
        meaning, exclude=have, min_trust=p._min_trust,
        limit=p._embedding.recall_facts + 3, margin=embeddings.RECALL_MARGIN,
        min_score=p._embedding.min_score)
    tids0 = [fid for fid, key in back.items() if key == target_key]
    out["target_in_lexical"] = bool(tids0) and tids0[0] in have
    out["by_meaning_pipe"] = [back.get(f["fact_id"], "?") for f in admits_h]
    # цель
    tids = [fid for fid, key in back.items() if key == target_key]
    if tids:
        tid = tids[0]
        out["target_in_sims"] = tid in meaning.sims
        if tid in meaning.sims:
            out["target_z"] = round(meaning.z(tid), 3)
            out["target_sim"] = round(meaning.sims[tid], 4)
            out["target_standout"] = meaning.standout(tid, embeddings.RECALL_MARGIN,
                                                      p._embedding.min_score)
            row = {"fact_id": tid}
            rows0 = p._retriever._rows([tid], None, p._min_trust)  # noqa: SLF001
            if rows0:
                row.update(rows0[0])
            out["target_names_fit"] = p._retriever.names_fit(query, row)
            out["target_admitted"] = any(f["fact_id"] == tid for f in admits)
    # причина-кандидат
    if not out["by_meaning"]:
        if zbar is not None and top and not meaning.standout(top[0][0], embeddings.RECALL_MARGIN,
                                                             p._embedding.min_score):
            out["reason"] = f"standout top-1 failed (z={meaning.z(top[0][0]):.3f} < {zbar:.3f})"
        else:
            out["reason"] = "by_meaning empty (eligible/trust/fits)"
    else:
        out["reason"] = "admitted"
    return out


def _row(p, fact_id: int, query: str) -> dict:
    row = {"fact_id": fact_id}
    rows = p._retriever._rows([fact_id], None, p._min_trust)  # noqa: SLF001
    row.update(rows[0]) if rows else None
    return row


def run(set_name: str, cmp_name: str | None) -> None:
    contrast = pp.load_contrast(set_name)
    db = Path(tempfile.gettempdir()) / f"enmean_{set_name}.db"
    for s in ("", "-shm", "-wal"):
        try:
            Path(str(db) + s).unlink()
        except OSError:
            pass
    p = pp.make_provider(db, "off", "on")
    back = fill_store(p, contrast)
    print(f"[{set_name}] векторы: {pp.catch_up_vectors(p)}")
    rows = []
    for pair in contrast.PAIRS:
        for query, target in pair["queries"]:
            rows.append(trace(p, back, query, target))
    from collections import Counter
    c = Counter(r.get("reason", "?") for r in rows)
    print(f"== {set_name}: {len(rows)} запросов ==")
    for reason, k in c.most_common():
        print(f"  {reason}: {k}")
    admitted = [r for r in rows if r.get("by_meaning")]
    print(f"  with admits: {len(admitted)}")
    pipe = [r for r in rows if r.get("by_meaning_pipe")]
    tgt_pipe = [r for r in rows if r["target"] in (r.get("by_meaning_pipe") or [])]
    rescue = [r for r in tgt_pipe if r.get("target_in_lexical") is False]
    print(f"  admits after exclude=lexical: {len(pipe)}; target among them: {len(tgt_pipe)}; "
          f"meaning-ONLY (target not in lexical): {len(rescue)} {[r['target'] for r in rescue]}")
    for r in rows[:4]:
        print("   sample:", json.dumps(r, ensure_ascii=False)[:400])
    # цели: сколько прошли по z
    with_tz = [r for r in rows if "target_z" in r]
    if with_tz:
        passed = [r for r in with_tz if r.get("target_standout")]
        print(f"  target standout passed: {len(passed)}/{len(with_tz)}")
        sample = sorted(with_tz, key=lambda r: -r["target_z"])[:5]
        for r in sample:
            print(f"    {r['target']:<24} z={r['target_z']:.3f} sim={r['target_sim']:.4f} "
                  f"standout={r['target_standout']} names_fit={r.get('target_names_fit')} "
                  f"admitted={r.get('target_admitted')} (z_bar={r.get('z_bar')})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="contrast_en")
    ap.add_argument("--cmp", default="contrast_ru")
    args = ap.parse_args()
    run(args.set, args.cmp)
    print()
    if args.cmp:
        run(args.cmp, None)