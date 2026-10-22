# -*- coding: utf-8 -*-
"""Диагностика wide: почему real cmax падает. brute-force косинус vs meaning.sims."""
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
import _wide_store as ws  # noqa: E402

hmod = importlib.import_module("hermes_plugin_holographic")

SET = "contrast_zh"
contrast = pp.load_contrast(SET)
db = Path(tempfile.gettempdir()) / f"admw_{SET}.db"
for s in ("", "-shm", "-wal"):
    try:
        Path(str(db) + s).unlink()
    except OSError:
        pass
p = pp.make_provider(db, "gate_only", "on")
for _key, (text, entities) in contrast.FACTS.items():
    p.handle_tool_call("fact_store", {"action": "add", "content": text, "entities": entities})
for _key, text, entities in ws.crowd_facts(contrast, SET.replace("contrast_", "")):
    p.handle_tool_call("fact_store", {"action": "add", "content": text, "entities": entities})
multi = pp.load_multilingual()
for _lg, facts in multi.CORPUS.items():
    for _k, (text, entities) in facts.items():
        p.handle_tool_call("fact_store", {"action": "add", "content": text, "entities": entities})
ok = pp.catch_up_vectors(p)
print("catch_up:", ok)

import sqlite3  # noqa: E402
import numpy as np  # noqa: E402
con = sqlite3.connect(str(db))
cur = con.cursor()
n_facts = cur.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
n_emb = cur.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0]
missing = [r for (r,) in cur.execute(
    "SELECT f.fact_id FROM facts f LEFT JOIN fact_embeddings e ON e.fact_id=f.fact_id "
    "WHERE e.fact_id IS NULL")]
print(f"facts={n_facts} emb={n_emb} missing={len(missing)} ids={missing[:30]}")
for fid in missing[:10]:
    c = cur.execute("SELECT content FROM facts WHERE fact_id=?", (fid,)).fetchone()[0]
    print(f"   #{fid}: {c[:70]}")
meta = dict(cur.execute("SELECT key, value FROM meta"))
print("meta:", meta)

# эмбеддинг запроса через тот же провайдер
queries = [(q, t) for pair in contrast.PAIRS for (q, t) in pair["queries"]]
vec_rows = {fid: np.frombuffer(v, dtype=np.float32)
            for fid, v in cur.execute("SELECT fact_id, vector FROM fact_embeddings")}
contents = {fid: c for fid, c in cur.execute("SELECT fact_id, content FROM facts")}


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n else v


for msg, tgt in queries[:6]:
    meaning = p._meaning_of(msg, before_turn=True)
    cmax = z = None
    if meaning and meaning.sims:
        best = max(meaning.sims, key=meaning.sims.get)
        cmax = float(meaning.sims[best])
        z = float(meaning.z(best))
    # brute force
    qv = norm(np.asarray(p._embedder.query(msg), dtype=np.float32))
    best_id, best_c, tgt_c, tgt_id = None, -1.0, None, None
    for fid, v in vec_rows.items():
        c = float(np.dot(qv, norm(v)))
        if c > best_c:
            best_id, best_c = fid, c
        if contents.get(fid) == tgt:
            tgt_c, tgt_id = c, fid
    print(f"\nq: {msg[:60]}")
    print(f"  target: {tgt[:60]}")
    print(f"  meaning cmax={cmax} z={z} best_id={best if meaning and meaning.sims else None}")
    print(f"  brute   cmax={best_c:.3f} best=#{best_id} content={contents.get(best_id, '')[:50]!r}")
    print(f"  target  cos={tgt_c} fact_id={tgt_id}")
    if meaning and meaning.sims and tgt_id is not None:
        print(f"  meaning.sims[target]={meaning.sims.get(tgt_id)}")
p.shutdown()
