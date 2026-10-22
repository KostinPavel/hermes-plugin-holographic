# -*- coding: utf-8 -*-
"""Свипы рычагов (пункты 1-2 плана), emb=on, 4 сета:

  gap — критерий top-gap δ поверх окна: δ 0.05 / 0.03 / 0.02 против базы (окно без вето);
  pf  — ширина семпотока embedding_prefetch_facts 1 / 2 / 3.

Запуск (из корня репо):
    PYTHONPATH=~/.hermes/hermes-agent HERMES_AGENT_REPO=~/.hermes/hermes-agent \
        python -I tools/_lever_sweep.py gap|pf
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_REPO / "tools"))

import provenance_probe as pp  # noqa: E402

SETS = ("contrast_ru", "contrast_en", "contrast_zh")
ALL = SETS + ("multi",)
MULTI_LANGS = "ar,bn,de,en,es,fr,hi,id,it,ja,ko,nl,pl,pt,ru,th,tr,zh".split(",")


def run_set(set_name: str, modes: list[str], extra: dict | None = None) -> list[dict]:
    if set_name == "multi":
        return pp.run_multilingual(MULTI_LANGS, modes, "on", pp.ENDPOINT, extra)
    return pp.run_contrast(set_name, modes, "on", pp.ENDPOINT, extra)


def agg(rows: list[dict]) -> dict:
    real = [r for r in rows if r["kind"] == "real"]
    chit = [r for r in rows if r["kind"] != "real"]
    return {
        "hit": sum(1 for r in real if r["hit"]),
        "n": len(real),
        "noise": sum(1 for r in chit if r["keys"]),
        "chit": len(chit),
        "wrong": sum(1 for r in real if r["wrong"]),
        "meaning": sum(1 for r in real for t in (r.get("sources") or {}).values() if "meaning" in t),
    }


def report(header: str, collected: list[tuple[str, str, dict]]) -> None:
    print(header)
    print("| вариант | сет | hit | шум | wrong5 | meaning |")
    print("|---|---|---|---|---|---|")
    for variant, set_name, a in collected:
        print(f"| {variant} | {set_name} | {a['hit']}/{a['n']} | {a['noise']}/{a['chit']} | "
              f"{a['wrong']} | {a['meaning']} |")


def phase_gap() -> None:
    modes = ["off", "gate_only", "topgap", "gate_gap03", "gate_gap02"]
    labels = {"off": "off", "gate_only": "окно 0.05 (дефолт)",
              "topgap": "окно+gap0.05", "gate_gap03": "окно+gap0.03", "gate_gap02": "окно+gap0.02"}
    out: list[tuple[str, str, dict]] = []
    for set_name in ALL:
        rows = run_set(set_name, modes)
        for mode in modes:
            out.append((labels[mode], set_name, agg([r for r in rows if r["mode"] == mode])))
    report("## gap-свип (emb=on)", out)


def phase_pf() -> None:
    modes = ["off", "gate_only"]
    out: list[tuple[str, str, dict]] = []
    for pf in (1, 2, 3):
        extra = None if pf == 1 else {"embedding_prefetch_facts": pf}
        for set_name in ALL:
            rows = run_set(set_name, modes, extra)
            for mode in modes:
                out.append((f"pf={pf} {mode}", set_name, agg([r for r in rows if r["mode"] == mode])))
    report("## recall_facts-свип (emb=on)", out)


if __name__ == "__main__":
    {"gap": phase_gap, "pf": phase_pf}[sys.argv[1]]()