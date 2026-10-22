# -*- coding: utf-8 -*-
"""Профилировка слоёв prefetch (base vs wide) + свип rerank_top_k (K=10/15/20/40).

Три режима (прод-код не меняется: патч только экземпляров провайдера в прогоне,
как в provenance_probe):

  profile — послойный тайминг prefetch, gate_only, emb=on, на base (~64 фактов)
            и wide (~620); по 75 запросам на сет (30 real + 45 chit) x 3 контраста.
            Слои: meaning_of (query embed :1237 + индекс), search (FTS+ngram+scan+blend),
            coverage, by_meaning, entity, rerank (HTTP :1235 + JSON-парс), store, other.
  ksweep  — качество (hit/шум/wrong5) + p50/p95 по каждому K на BASE-базе.
  widelat — контроль латентности (и качества) K на WIDE-базе.

Запуск (из корня репозитория):
    PYTHONPATH=~/.hermes/hermes-agent HERMES_AGENT_REPO=~/.hermes/hermes-agent \
      /home/pavel/.hermes/hermes-agent/venv/bin/python -I tools/_lat_profile.py profile
    ... -I tools/_lat_profile.py ksweep --ks 10,15,20,40
    ... -I tools/_lat_profile.py widelat --ks 15,40

Промежуточные данные: results/_lat_<mode>.json(.jsonl); агрегат печатается в stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_REPO / "tools"))

import provenance_probe as pp  # noqa: E402
import _wide_store as ws  # noqa: E402
import hermes_plugin_holographic as _H  # noqa: E402

SETS = ["contrast_ru", "contrast_en", "contrast_zh"]
RERANK = _H.rerank  # модуль клиента реранка (патчим _OPENER/json только в profile)


# ---------------------------------------------------------------- тайминг слоёв

class Recorder:
    """Накопитель миллисекунд по слоям внутри ОДНОГО prefetch."""

    def __init__(self) -> None:
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self.stats: dict = {}

    def reset(self) -> None:
        self.t.clear()
        self.n.clear()
        self.stats.clear()

    def add(self, layer: str, ms: float) -> None:
        self.t[layer] = self.t.get(layer, 0.0) + ms
        self.n[layer] = self.n.get(layer, 0) + 1


REC = Recorder()


def _wrap(obj, name: str, layer: str) -> None:
    """Обернуть метод объекта, накапливая время в REC."""
    orig = getattr(obj, name, None)
    if orig is None or not callable(orig):
        return

    def call(*a, **kw):
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        REC.add(layer, (time.perf_counter() - t0) * 1000.0)
        return out

    setattr(obj, name, call)


class _RespProxy:
    """Контекст response: время от open() до закрытия = полный HTTP (:1235)."""

    def __init__(self, resp, t0: float):
        self._r = resp
        self._t0 = t0

    def __enter__(self):
        self._r.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return self._r.__exit__(*exc)
        finally:
            REC.add("rerank_http", (time.perf_counter() - self._t0) * 1000.0)

    def read(self, *a, **kw):
        return self._r.read(*a, **kw)

    def __getattr__(self, item):
        return getattr(self._r, item)


class _OpenerProxy:
    def __init__(self, orig):
        self._o = orig

    def open(self, *a, **kw):
        t0 = time.perf_counter()
        return _RespProxy(self._o.open(*a, **kw), t0)


class _JsonProxy:
    """json модуля реранка: считаем только loads (парс ответа)."""

    def __init__(self, real):
        self._j = real

    def loads(self, *a, **kw):
        t0 = time.perf_counter()
        out = self._j.loads(*a, **kw)
        REC.add("rerank_parse", (time.perf_counter() - t0) * 1000.0)
        return out

    def __getattr__(self, item):
        return getattr(self._j, item)


_PATCHED = {"open": False}


def install_rerank_probe() -> None:
    """Патч модуля rerank: HTTP-слой и JSON-парс отдельными слоями."""
    if _PATCHED["open"]:
        return
    RERANK._OPENER = _OpenerProxy(RERANK._OPENER)
    RERANK.json = _JsonProxy(RERANK.json)
    _PATCHED["open"] = True


def instrument(p) -> None:
    """Патч экземпляра провайдера: слои prefetch."""
    _wrap(p, "_meaning_of", "meaning_of")
    _wrap(p._embedder, "query", "embed_query")
    _wrap(p._retriever, "meaning", "meaning_index")
    _wrap(p._retriever, "search", "search")
    _wrap(p._retriever, "relevant", "coverage")
    _wrap(p._retriever, "by_meaning", "by_meaning")
    _wrap(p._retriever, "entity_facts", "entity")
    _wrap(p._store, "mentioned_entity_ids", "store")
    _wrap(p._store, "current_sources", "store")
    _wrap(p._store, "origins", "store")

    # rerank: суммарно + счётчики документов
    if p._rescorer is not None:
        orig_scores = p._rescorer.scores

        def scores(query, documents):
            REC.stats["docs_in"] = len(documents)
            REC.stats["docs_sent"] = min(int(p._rescorer.settings.top_k), len(documents))
            t0 = time.perf_counter()
            out = orig_scores(query, documents)
            REC.add("rerank", (time.perf_counter() - t0) * 1000.0)
            REC.stats["rerank_ok"] = out is not None
            if out is not None:
                REC.stats["ranked_n"] = len(out)
            return out

        p._rescorer.scores = scores


# верхнеуровневые слои: сумма которых вычитается из total для «прочего»
TOP_LAYERS = ("meaning_of", "search", "coverage", "by_meaning", "entity", "rerank", "store")
ALL_LAYERS = ("meaning_of", "embed_query", "meaning_index", "search", "coverage",
              "by_meaning", "entity", "rerank", "rerank_http", "rerank_parse", "store", "other")


def _unlink(db: Path) -> None:
    for suffix in ("", "-shm", "-wal"):
        try:
            Path(str(db) + suffix).unlink()
        except OSError:
            pass


def percentile(xs: list[float], q: float) -> float:
    return ws.percentile(xs, q)


# ---------------------------------------------------------------- profile

def build_store(p, set_name: str, wide: bool) -> tuple[dict[int, str], int]:
    """Факты сета (+ крауд + CORPUS при wide), как в _wide_store.run."""
    contrast = pp.load_contrast(set_name)
    lang = set_name.replace("contrast_", "")
    back: dict[int, str] = {}
    n = 0

    def add(text: str, entities, key: str) -> None:
        nonlocal n
        r = json.loads(p.handle_tool_call(
            "fact_store", {"action": "add", "content": text, "entities": entities}))
        back[int(r["fact_id"])] = key
        n += 1

    for key, (text, entities) in contrast.FACTS.items():
        add(text, entities, key)
    if wide:
        for key, text, entities in ws.crowd_facts(contrast, lang):
            add(text, entities, key)
        multi = pp.load_multilingual()
        for lg, facts in multi.CORPUS.items():
            for key, (text, entities) in facts.items():
                add(text, entities, f"multi_{lg}_{key}")
    return back, n


def profile_run(set_name: str, wide: bool) -> list[dict]:
    size = "wide" if wide else "base"
    db = Path(tempfile.gettempdir()) / f"lat_{set_name}_{size}.db"
    _unlink(db)
    p = pp.make_provider(db, "gate_only", "on")
    back, n_facts = build_store(p, set_name, wide)
    t0 = time.perf_counter()
    assert pp.catch_up_vectors(p), "векторы не готовы"
    embed_s = round(time.perf_counter() - t0, 1)
    instrument(p)
    contrast = pp.load_contrast(set_name)

    rows: list[dict] = []
    queries = [("real", q) for pair in contrast.PAIRS for (q, _t) in pair["queries"]]
    queries += [("chit", q) for (q, _cat) in contrast.CHITCHAT]
    for i, (kind, query) in enumerate(queries):
        p.on_session_switch(f"lat{i}", reset=True)
        REC.reset()
        t0 = time.perf_counter()
        block = p.prefetch(query)
        total = (time.perf_counter() - t0) * 1000.0
        layers = {k: round(v, 2) for k, v in REC.t.items()}
        used = sum(layers.get(k, 0.0) for k in TOP_LAYERS)
        rows.append({
            "set": set_name, "size": size, "kind": kind, "q": query,
            "total_ms": round(total, 2),
            "layers": layers,
            "other": round(max(0.0, total - used), 2),
            "docs_in": REC.stats.get("docs_in", 0),
            "docs_sent": REC.stats.get("docs_sent", 0),
            "rerank_ok": REC.stats.get("rerank_ok", None),
            "meaning": REC.t.get("meaning_of", 0.0) > 0.05,
            "ids": pp.parse_ids(block),
            "pool_hint": len(REC.t),
            "calls": {k: REC.n.get(k, 0) for k in REC.n},
        })
    p.shutdown()
    out = _REPO / "results" / f"_lat_profile_{set_name}_{size}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [{set_name} {size}] facts={n_facts} embed={embed_s}s -> {out}", flush=True)
    for r in rows:
        r["_facts"] = n_facts
        r["_embed_s"] = embed_s
    return rows


def layer_stats(rows: list[dict]) -> list[dict]:
    """p50/p95 слоя по запросам + средние по когорте самых тяжёлых 5% total."""
    totals = [r["total_ms"] for r in rows]
    p95_cut = percentile(totals, 0.95)
    cohort = [r for r in rows if r["total_ms"] >= p95_cut]
    out = []
    for layer in ALL_LAYERS:
        vals = [(r["layers"].get(layer, 0.0) if layer != "other" else r["other"]) for r in rows]
        if not any(vals):
            continue
        cv = [(r["layers"].get(layer, 0.0) if layer != "other" else r["other"]) for r in cohort]
        mean_cohort = sum(cv) / len(cv) if cv else 0.0
        mean_total_cohort = sum(r["total_ms"] for r in cohort) / len(cohort) if cohort else 0.0
        out.append({
            "layer": layer,
            "p50": round(percentile(vals, 0.5), 1),
            "p95": round(percentile(vals, 0.95), 1),
            "mean_p95_cohort": round(mean_cohort, 1),
            "share_p95_cohort": round(100.0 * mean_cohort / mean_total_cohort, 1) if mean_total_cohort else 0.0,
        })
    return out


def cmd_profile(args) -> None:
    install_rerank_probe()
    all_rows: list[dict] = []
    sizes = ["base", "wide"] if args.size == "both" else [args.size]
    for set_name in args.sets.split(","):
        for size in sizes:
            all_rows += profile_run(set_name.strip(), size == "wide")

    summary = {}
    for size in sizes:
        rows = [r for r in all_rows if r["size"] == size]
        totals = [r["total_ms"] for r in rows]
        stats = layer_stats(rows)
        summary[size] = {
            "n": len(rows),
            "facts": rows[0]["_facts"],
            "embed_s": rows[0]["_embed_s"],
            "total_p50": round(percentile(totals, 0.5), 1),
            "total_p95": round(percentile(totals, 0.95), 1),
            "total_max": round(max(totals), 1),
            "rerank_docs": round(sum(r["docs_sent"] for r in rows) / max(1, len(rows)), 1),
            "meaning_calls": sum(1 for r in rows if r["meaning"]),
            "layers": stats,
        }
        print(f"\n=== {size}: {len(rows)} запросов, facts={summary[size]['facts']}, "
              f"total p50/p95 = {summary[size]['total_p50']}/{summary[size]['total_p95']} мс, "
              f"max {summary[size]['total_max']} мс, embed базы {summary[size]['embed_s']} с, "
              f"документов в CE (ср.) {summary[size]['rerank_docs']}, "
              f"вызовов embed запроса {summary[size]['meaning_calls']}/{len(rows)}")
        print(f"| {'слой':<14} | p50 мс | p95 мс | когорта p95: ср. мс | доля |")
        print("|---|---|---|---|---|")
        for s in stats:
            print(f"| {s['layer']:<14} | {s['p50']:>6} | {s['p95']:>6} | "
                  f"{s['mean_p95_cohort']:>18} | {s['share_p95_cohort']:>4}% |")
    out = _REPO / "results" / "_lat_profile_summary.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"\nсводка -> {out}")


# ---------------------------------------------------------------- ksweep / widelat

_ORIG_MAKE = pp.make_provider
_K: dict = {"value": None}
LAST: dict = {}  # последний провайдер: читаем счётчики calls/failures реранка после прогона


def _make_with_k(db, mode, emb, extra=None):
    merged = dict(extra or {})
    if _K["value"]:
        merged["rerank_top_k"] = int(_K["value"])
    p = _ORIG_MAKE(db, mode, emb, merged)
    LAST["p"] = p
    return p


def rerank_counters() -> dict:
    p = LAST.get("p")
    r = getattr(p, "_rescorer", None)
    if r is None:
        return {"calls": None, "failures": None}
    return {"calls": r.calls, "failures": r.failures}


pp.make_provider = _make_with_k  # _wide_store.run вызывает pp.make_provider по ссылке


def _agg(run_rows: list[dict]) -> dict:
    return {
        "facts": [r["facts"] for r in run_rows],
        "hit": sum(r["hit"] for r in run_rows),
        "n": sum(r["n"] for r in run_rows),
        "noise": sum(r["noise"] for r in run_rows),
        "chit": sum(r["chit"] for r in run_rows),
        "wrong": sum(r["wrong"] for r in run_rows),
        "pool_p50": [r["pool_p50"] for r in run_rows],
        "lat_p50": [r["lat_p50"] for r in run_rows],
        "lat_p95": [r["lat_p95"] for r in run_rows],
        "misses": {m for r in run_rows for m in r["misses"]},
        "rows": run_rows,
    }


def _dump(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1, default=sorted)
    print(f"-> {path}", flush=True)


def cmd_ksweep(args) -> None:
    ks = [int(k) for k in args.ks.split(",")]
    result = {}
    for k in ks:
        _K["value"] = k
        result[str(k)] = {}
        for set_name in SETS:
            row = ws.run(set_name, False, "gate_only")
            row["rerank"] = rerank_counters()
            result[str(k)][set_name] = row
            print(f"K={k:>2} {set_name}: hit {row['hit']}/{row['n']} noise {row['noise']}/{row['chit']} "
                  f"wrong {row['wrong']} lat {row['lat_p50']}/{row['lat_p95']} мс "
                  f"pool p50 {row['pool_p50']:.0f} embed {row['embed_s']}с "
                  f"rerank calls/fail {row['rerank']['calls']}/{row['rerank']['failures']}", flush=True)
    _K["value"] = None
    print("\n| K | Σhit/90 | Σшум/135 | Σwrong5 | lat p50/p95 (по сетам, мс) | pool p50 |")
    print("|---|---|---|---|---|---|")
    agg = {}
    for k in ks:
        a = _agg([result[str(k)][s] for s in SETS])
        agg[str(k)] = a
        lat = "/".join(str(x) for x in a["lat_p50"]) + " | " + "/".join(str(x) for x in a["lat_p95"])
        print(f"| {k} | {a['hit']}/{a['n']} | {a['noise']}/{a['chit']} | {a['wrong']} | "
              f"{lat} | {'/'.join(f'{x:.0f}' for x in a['pool_p50'])} |")
    _dump(_REPO / "results" / "_lat_ksweep.json", {"per_k": result, "agg": {k: {kk: vv for kk, vv in a.items()
                                                                     if kk != 'rows'} for k, a in agg.items()}})


def cmd_widelat(args) -> None:
    ks = [int(k) for k in args.ks.split(",")]
    result = {}
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    for k in ks:
        _K["value"] = k
        result[str(k)] = {}
        for set_name in sets:
            row = ws.run(set_name, True, "gate_only")
            row["rerank"] = rerank_counters()
            result[str(k)][set_name] = row
            print(f"wide K={k:>2} {set_name}: facts {row['facts']} hit {row['hit']}/{row['n']} "
                  f"noise {row['noise']}/{row['chit']} wrong {row['wrong']} "
                  f"lat {row['lat_p50']}/{row['lat_p95']} мс pool p50 {row['pool_p50']:.0f} "
                  f"bind40 {row['bind40']} embed {row['embed_s']}с "
                  f"rerank calls/fail {row['rerank']['calls']}/{row['rerank']['failures']}", flush=True)
    _K["value"] = None
    print(f"\n| K | база | Σhit/{30 * len(sets)} | Σшум/{45 * len(sets)} | Σwrong5 | lat p50/p95 (по сетам, мс) | pool p50 | >=40 |")
    print("|---|---|---|---|---|---|---|---|")
    agg = {}
    for k in ks:
        a = _agg([result[str(k)][s] for s in sets])
        agg[str(k)] = a
        lat = "/".join(str(x) for x in a["lat_p50"]) + " | " + "/".join(str(x) for x in a["lat_p95"])
        pools = [r["pool_p50"] for r in a["rows"]]
        bind = [r["bind40"] for r in a["rows"]]
        print(f"| {k} | wide | {a['hit']}/{a['n']} | {a['noise']}/{a['chit']} | {a['wrong']} | "
              f"{lat} | {'/'.join(f'{x:.0f}' for x in pools)} | {'/'.join(str(x) for x in bind)} |")
    _dump(_REPO / "results" / "_lat_widelat.json", {"per_k": result,
                                                    "agg": {k: {kk: vv for kk, vv in a.items() if kk != 'rows'}
                                                            for k, a in agg.items()}})


def main(argv) -> None:
    ap = argparse.ArgumentParser(description="Профилировка слоёв prefetch + свип rerank_top_k")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("profile")
    pr.add_argument("--sets", default=",".join(SETS))
    pr.add_argument("--size", default="both", choices=["base", "wide", "both"])
    pr.set_defaults(func=cmd_profile)
    kw = sub.add_parser("ksweep")
    kw.add_argument("--ks", default="10,15,20,40")
    kw.set_defaults(func=cmd_ksweep)
    wl = sub.add_parser("widelat")
    wl.add_argument("--ks", default="15,40")
    wl.add_argument("--sets", default=",".join(SETS))
    wl.set_defaults(func=cmd_widelat)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])