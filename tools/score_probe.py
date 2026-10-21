# -*- coding: utf-8 -*-
"""Какая форма гейта имеет смысл на sigmoid: отношение (margin), разрыв (gap)
или абсолютный порог? Собирает (best, second) по 30 реальным и 15 chit-chat
запросам, считает обе метрики в нормализованной шкале и печатает разделяемость.

    HERMES_AGENT_REPO=... python tools/score_probe.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "contrast_off", Path(__file__).resolve().parent / "contrast_off.py")
co = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(co)

from hermes_plugin_holographic import HolographicMemoryProvider  # noqa: E402
from hermes_plugin_holographic import rerank  # noqa: E402

# Запись всех ответов эндпоинта (сырые логиты)
_records: list[tuple[str, list[float]]] = []
_orig = rerank.Rescorer.scores


def _rec(self, query, documents):
    out = _orig(self, query, documents)
    if out:
        _records.append((query, [s for _, s in out]))
    return out


rerank.Rescorer.scores = _rec


def _sig(x: float) -> float:
    if x >= 0.0:
        import math
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    import math
    z = math.exp(x)
    return z / (1.0 + z)


def main() -> None:
    import math
    db = Path(math.gettempdir()) if False else Path(co.tempfile.gettempdir()) / "score_probe.db"
    for s in ("", "-shm", "-wal"):
        try:
            Path(str(db) + s).unlink()
        except OSError:
            pass
    p = co._provider(db, rerank="on", rerank_mode="gate_reads_score",
                     rerank_weight="0.0", rerank_url=co.RERANK_URL,
                     rerank_model="local-reranker")
    co._fill(p)
    chit = {q for q, _ in co.contrast.CHITCHAT}
    queries = [q for pair in co.contrast.PAIRS for q, _ in pair["queries"]]
    for i, q in enumerate(queries + [q for q, _ in co.contrast.CHITCHAT]):
        p.on_session_switch(f"probe_{i}", reset=True)
        p.prefetch(q)
    p.shutdown()

    rows = []
    for query, scores in list(_records):
        if not scores:
            continue
        norm = [_sig(s) for s in scores]  # логиты -> (0,1)
        best, second = (norm[0], norm[1]) if len(norm) > 1 else (norm[0], None)
        label = "chit" if query in chit else "real"
        rows.append((label, query, best, second))

    print(f"{'вид':<5} {'best':>7} {'second':>7} {'gap':>7} {'ratio':>7}  запрос")
    reals = [(b, s, q) for l, q, b, s in rows if l == "real" and s is not None]
    chits = [(b, s, q) for l, q, b, s in rows if l == "chit" and s is not None]
    for b, s, q in sorted(reals, key=lambda x: x[0] - x[1]):
        print(f"{'real':<5} {b:>7.4f} {s:>7.4f} {b - s:>7.4f} {b / s:>7.3f}  {q[:48]}")
    for b, s, q in sorted(chits, key=lambda x: x[0] - x[1]):
        print(f"{'chit':<5} {b:>7.4f} {s:>7.4f} {b - s:>7.4f} {b / s:>7.3f}  {q[:48]}")

    def sep(vals, name):
        if not vals:
            return
        print(f"  {name}: min={min(vals):.4f} max={max(vals):.4f} n={len(vals)}")

    print("\n=== разделяемость ===")
    sep([b - s for b, s, _ in reals], "gap real")
    sep([b - s for b, s, _ in chits], "gap chit")
    sep([b / s for b, s, _ in reals], "ratio real")
    sep([b / s for b, s, _ in chits], "ratio chit")
    rg = [b - s for b, s, _ in reals]
    cg = [b - s for b, s, _ in chits]
    rr = [b / s for b, s, _ in reals]
    cr = [b / s for b, s, _ in chits]
    if rg and cg:
        print(f"  GAP: порог, разделяющий max(chit) и min(real): {max(cg):.4f} .. {min(rg):.4f} "
              f"({'РАЗДЕЛЯЕТ' if max(cg) < min(rg) else 'НЕ разделяет'})")
    if rr and cr:
        print(f"  RATIO: порог, разделяющий max(chit) и min(real): {max(cr):.4f} .. {min(rr):.4f} "
              f"({'РАЗДЕЛЯЕТ' if max(cr) < min(rr) else 'НЕ разделяет'})")


if __name__ == "__main__":
    main()