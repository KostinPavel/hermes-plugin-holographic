# -*- coding: utf-8 -*-
"""Сравнение гипотезы «реранкер либо фильтр как фолбэк»:
off vs reorder vs gate@1.0 (чистый admit-all) vs override (топ-3 вопреки coverage).

    HERMES_AGENT_REPO=... CONTRAST_SET=contrast_ru python tools/variant_run.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "contrast_off", Path(__file__).resolve().parent / "contrast_off.py")
co = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(co)

MODES = ["off", "reorder_0", "gate_m100", "override_0"]

print(f"{'режим':<11} {'пар':>3} {'MISS':>5} {'WRONG5':>7} {'тиш':>6} {'шум':>5} {'p50мс':>7}")
for m in MODES:
    r = co.run("on", m)
    noise = (r["total"] - r["quiet"]) / r["total"] if r["total"] else 0.0
    print(f"{r['mode']:<11} {r['pair_pass']:>3} {r['miss']:>5} {r['partner_top5']:>7} "
          f"{r['quiet']}/{r['total']:>2} {noise:>5.0%} {r['p50_ms']:>7.1f}")