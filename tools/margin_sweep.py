# -*- coding: utf-8 -*-
"""Свип rerank_margin для gate_reads_score после sigmoid-фикса (contrast-ru).

Отвечает на вопрос: какой порог на шкале (0,1) разделяет chit-chat (flat,
ratio ~1.0x) и реальные запросы (уверенный топ, ratio ~1.0x-1.4x)?

    HERMES_AGENT_REPO=... python tools/margin_sweep.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "contrast_off", Path(__file__).resolve().parent / "contrast_off.py")
co = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(co)

MODES = ["off", "gate_0", "gate_m12", "gate_m11", "gate_m105", "gate_m102"]

print(f"{'режим':<11} {'пар':>3} {'MISS':>5} {'WRONG5':>7} {'тиш':>5} {'шум':>5} {'p50мс':>7}")
for m in MODES:
    r = co.run("on", m)
    noise = (r["total"] - r["quiet"]) / r["total"] if r["total"] else 0.0
    print(f"{r['mode']:<11} {r['pair_pass']:>3} {r['miss']:>5} {r['partner_top5']:>7} "
          f"{r['quiet']}/{r['total']:>2} {noise:>5.0%} {r['p50_ms']:>7.1f}")