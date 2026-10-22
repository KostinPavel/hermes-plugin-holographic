# -*- coding: utf-8 -*-
"""Смешанный vs раздельный магазин: факты en и ru в одном store против двух отдельных.
Метрики: hit (цель в top-5), шум (chit вернул факты). emb=on, gate_only.

Запуск:
    PYTHONPATH=~/.hermes/hermes-agent HERMES_AGENT_REPO=~/.hermes/hermes-agent \
        python -I tools/_mixed_store.py
"""
from __future__ import annotations

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

import provenance_probe as pp  # noqa: E402


def fill(p, mods):
    back = {}
    for mod in mods:
        for key, (text, entities) in mod.FACTS.items():
            r = json.loads(p.handle_tool_call(
                "fact_store", {"action": "add", "content": text, "entities": entities}))
            back[int(r["fact_id"])] = (mod.__name__.replace("contrast_", ""), key)
    return back


def measure(mods, label: str) -> None:
    db = Path(tempfile.gettempdir()) / f"mixed_{label}.db"
    for s in ("", "-shm", "-wal"):
        try:
            Path(str(db) + s).unlink()
        except OSError:
            pass
    p = pp.make_provider(db, "gate_only", "on")
    back = fill(p, mods)
    assert pp.catch_up_vectors(p), "векторы не готовы"
    probe = pp.Probe(p)
    hit = n = noise = chit = 0
    per_lang = {}
    for mod in mods:
        lang = mod.__name__.replace("contrast_", "")
        lh = ln = nc = cc = 0
        for pi, pair in enumerate(mod.PAIRS):
            for i, (query, target) in enumerate(pair["queries"]):
                p.on_session_switch(f"{label}_{lang}_{pi}_{i}", reset=True)
                block, _ctx = probe.prefetch(query)
                found = pp.parse_ids(block)
                keys = {back[f][1] for f in found if f in back}
                ln += 1
                lh += target in keys
        for i, (query, _cat) in enumerate(mod.CHITCHAT):
            p.on_session_switch(f"{label}_{lang}_q{i}", reset=True)
            block, _ctx = probe.prefetch(query)
            found = pp.parse_ids(block)
            keys = {back[f][1] for f in found if f in back}
            # шум считаем по ФАКТАМ ЭТОГО языка (чужой язык в chit-выдаче — тоже шум, но считаем общий)
            cc += 1
            nc += bool(found)
        per_lang[lang] = (lh, ln, nc, cc)
        hit += lh
        n += ln
        noise += nc
        chit += cc
    p.shutdown()
    detail = " ".join(f"{lg}: {h}/{nn} hit, {nc_}/{cc} noise" for lg, (h, nn, nc_, cc) in sorted(per_lang.items()))
    print(f"| {label} | {hit}/{n} | {noise}/{chit} | {detail} |")


print("| вариант | Σhit | Σшум | по языкам |")
print("|---|---|---|---|")
en = pp.load_contrast("contrast_en")
ru = pp.load_contrast("contrast_ru")
measure([en], "раздельно en")
measure([ru], "раздельно ru")
measure([en, ru], "смешанно en+ru")