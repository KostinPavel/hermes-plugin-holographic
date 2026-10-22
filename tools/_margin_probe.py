# -*- coding: utf-8 -*-
"""Как выглядят отвергнутые margin-вето запросы при rerank_margin=1.05:
топ-2 после sigmoid, реальный ли это запрос, попала ли цель."""
from __future__ import annotations
import importlib.util, json, os, sys, tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_spec = importlib.util.spec_from_file_location(
    "hermes_plugin_holographic", _REPO / "__init__.py", submodule_search_locations=[str(_REPO)])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hermes_plugin_holographic"] = _mod
_spec.loader.exec_module(_mod)
from hermes_plugin_holographic import HolographicMemoryProvider  # noqa: E402
from hermes_plugin_holographic import rerank as _rerank  # noqa: E402

_SET = os.environ.get("CONTRAST_SET", "contrast_zh")
cs = importlib.util.spec_from_file_location(_SET, _REPO / "tools" / f"{_SET}.py")
contrast = importlib.util.module_from_spec(cs)
cs.loader.exec_module(contrast)

p = HolographicMemoryProvider(config={
    "db_path": tempfile.mktemp(suffix=".db"), "hrr_dim": 1024,
    "rerank": "on", "rerank_mode": "gate_reads_score", "rerank_weight": "0.0",
    "rerank_margin": os.environ.get("MARGIN", "1.05"),
    "rerank_url": "http://127.0.0.1:1235/v1/rerank", "rerank_model": "probe"})
p.initialize(session_id="m")

import traceback as _tb
def _rf(e):  # показать полный traceback вместо "kind only"
    _tb.print_exception(type(e), e, e.__traceback__)
p._recall_failed = _rf

for text, ent in contrast.FACTS.values():
    p.handle_tool_call("fact_store", {"action": "add", "content": text, "entities": ent})

records = []
orig = p._rescorer.scores
def spy(query, docs):
    ranked = orig(query, docs)
    if ranked and len(ranked) > 1:
        norm = _rerank.normalized(ranked)
        records.append({"q": query, "n_docs": len(docs),
                        "best": round(norm[0], 4), "second": round(norm[1], 4),
                        "ratio": round(norm[0] / norm[1], 3) if norm[1] else None,
                        "gap": round(norm[0] - norm[1], 4)})
    elif ranked:
        records.append({"q": query, "n_docs": len(docs), "best": round(_rerank.normalized(ranked)[0], 4),
                        "second": None, "ratio": None, "gap": None})
    return ranked
p._rescorer.scores = spy

MARGIN = float(os.environ.get("MARGIN", "1.05"))
rows = [("real", q, contrast.FACTS[t][0]) for pr in contrast.PAIRS for (q, t) in pr["queries"]]
rows += [("chit", q, None) for q, _cat in contrast.CHITCHAT]
rejected, passed = [], []
for i, (kind, query, target) in enumerate(rows):
    p.on_session_switch(f"m{i}", reset=True)
    before = len(records)
    block = p.prefetch(query)
    rec = records[before] if len(records) > before else None
    if rec is None:
        continue
    row = {**rec, "kind": kind, "recall": bool(block),
           "target_in": bool(target) and target in (block or ""),
           "veto": rec["second"] is not None and rec["best"] < rec["second"] * MARGIN}
    (rejected if row["veto"] else passed).append(row)

print(f"=== {_SET}, margin={MARGIN}: вето {len(rejected)} из {len(rows)}")
print(f"{'kind':5} {'best':>7} {'second':>7} {'gap':>7} {'ratio':>6} veto recall tgt_in  query")
for r in sorted(rejected + passed, key=lambda x: (x["kind"], -(x["gap"] or 0))):
    print(f"{r['kind']:5} {r['best']:7.4f} {str(r['second']):>7} {str(r['gap']):>7} {str(r['ratio']):>6} "
          f"{'V' if r['veto'] else '.':4} {str(r['recall'])[0]:5} {str(r['target_in'])[0]:6} {r['q']!r}")
p.shutdown()