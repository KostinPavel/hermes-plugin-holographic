"""Benchmark the five rescoring modes (RFC #9) over the multilingual set.

Measures, per mode and per language: recall@5 (the ground-truth fact in the top-5 before the turn), wrong-entity
hits (a fact about another person in the top-5 when the query names one), and the noise on questions that must recall
nothing. A markdown table comes out — the shape itpartypattaya asked for in issue #9.

Quality uses a mock rerank endpoint (the order it answers with is the model's ranking). Latency is measured against a
real llama-server with a real reranker (--rerank-url, see --endpoint): 5 modes x N models, run on a CPU box, not in
the CI.

    PYTHONPATH=~/.hermes/hermes-agent python -I tools/benchmark_rerank.py                # quality, mock
    PYTHONPATH=~/.hermes/hermes-agent python -I tools/benchmark_rerank.py --endpoint=http://127.0.0.1:8080/v1/rerank

Modes: off, reorder_only, reorder_blend, gate_only, gate_blend (the two axes of issue #9: where the cross-encoder's
rank sits — before the coverage gate or as the gate — and whether it blends with the blend score or orders only)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# The package tree loads as hermes_plugin_holographic from the checkout (the CI's directory name rule: a valid
# Python identifier). hermes-agent's checkout goes on sys.path first (its `agent`, `tools`, `hermes_cli`).
import importlib.util
import os

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "hermes_plugin_holographic", _REPO / "__init__.py", submodule_search_locations=[str(_REPO)])
assert _spec and _spec.loader
if "hermes_plugin_holographic" not in sys.modules:
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["hermes_plugin_holographic"] = _mod
    _spec.loader.exec_module(_mod)

from hermes_plugin_holographic import HolographicMemoryProvider  # noqa: E402
from hermes_plugin_holographic import rerank  # noqa: E402

# The set lives in the test module (its CORPUS / QUERIES / PROBES are the benchmark's data). Load it directly: `tests`
# on sys.path would resolve to hermes-agent's own tests/ first.
assert _spec and _spec.loader
_data_spec = importlib.util.spec_from_file_location(
    "holo_lang_data", _REPO / "tests" / "test_holographic_languages.py")
assert _data_spec and _data_spec.loader
data = importlib.util.module_from_spec(_data_spec)
_data_spec.loader.exec_module(data)

# mode -> (rerank_mode, rerank_weight)
MODES = {
    "off": ("off", "0.0"),
    "reorder_only": ("reorder_before_gate", "0.0"),
    "reorder_blend": ("reorder_before_gate", "0.5"),
    "gate_only": ("gate_reads_score", "0.0"),
    "gate_blend": ("gate_reads_score", "0.5"),
}


class _Response:
    def __init__(self, body: bytes):
        self.body, self.at = body, 0

    def read(self, n=-1):
        piece = self.body[self.at:] if n is None or n < 0 else self.body[self.at:self.at + n]
        self.at += len(piece)
        return piece

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class MockRerank:
    """Stands in for rerank._OPENER: answers in a (mock) ranker's order. A decent ranker by default — a confident top
    (1.0) over a low tail — so the gate modes stay open on relevant queries; `--reverse` is the worst case. Chit-chat
    noise and honest latency come only from a real endpoint (--endpoint); the mock cannot tell a relevant query from a
    greeting, so its gate-mode noise column is not meaningful."""

    def __init__(self, reverse: bool = False):
        self.requests = 0
        self.reverse = reverse  # a ranker that inverts the order (worst case)

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests += 1
        n = len(body["documents"])
        order = list(reversed(range(n))) if self.reverse else list(range(n))
        top_n = min(body.get("top_n", n), n)
        results = [{"index": i, "relevance_score": 1.0 if rank == 0 else 0.5 - rank * 0.01}
                   for rank, i in enumerate(order[:top_n])]
        return _Response(json.dumps({"results": results}).encode())


def _provider(db_path: str, **config) -> HolographicMemoryProvider:
    p = HolographicMemoryProvider(config={"db_path": db_path, "hrr_dim": 1024, **config})
    p.initialize(session_id="bench")
    return p


def _fill(p, lang: str) -> dict:
    """The language's facts in; returns fact_id -> key."""
    back = {}
    for key, (text, entities) in data.CORPUS[lang].items():
        r = json.loads(p.handle_tool_call("fact_store", {"action": "add", "content": text, "entities": entities}))
        back[r["fact_id"]] = key
    return back


def _inject(p, query: str) -> set[str]:
    """The keys injected before the turn (prefetch), or the empty set."""
    block = p.prefetch(query)
    if not block:
        return set()
    # lines after the header are "- #<id> [trust] <text>"
    keys = set()
    for line in block.splitlines()[1:]:
        try:
            keys.add(line.split()[1][1:])  # the fact_id
        except (IndexError, ValueError):
            pass
    return keys


def _percentile(times: list[float], q: float) -> float:
    if not times:
        return 0.0
    idx = min(len(times) - 1, int(round(q * (len(times) - 1))))
    return sorted(times)[idx]


def run(langs: list[str], mode: str, endpoint_url: str | None, reverse: bool) -> dict:
    """One mode over the given languages: recall@5, wrong-entity hits, noise, latency p50/p95."""
    rerank_mode, weight = MODES[mode]
    config = {"rerank": "on" if rerank_mode != "off" else "off"}
    if rerank_mode != "off":
        config.update({"rerank_mode": rerank_mode, "rerank_weight": weight,
                       "rerank_model": "bench-reranker"})
        config["rerank_url"] = endpoint_url or "https://rerank.example.net/v1/rerank"

    # Mock: the order the fake ranker answers with is what the modes see.
    mock = MockRerank(reverse=reverse)
    real_opener = rerank._OPENER
    if not endpoint_url:
        rerank._OPENER = mock

    hit5 = asked = wrong = quiet = negatives = 0
    times: list[float] = []
    failures = 0  # ranked is None: fallbacks to the coverage order (counted in rerank.Rescorer)
    try:
        for li, lang in enumerate(langs):
            p = _provider(f"/tmp/bench_{lang}_{mode}.db", **config)
            back = _fill(p, lang)
            for i, (query, relevant) in enumerate(data.QUERIES[lang]):
                p.on_session_switch(f"bench{li}_{i}", reset=True)
                t0 = time.perf_counter()
                injected = _inject(p, query)
                times.append(time.perf_counter() - t0)
                found_keys = {back[int(k)] for k in injected if int(k) in back}
                if relevant:
                    asked += 1
                    hit5 += bool(found_keys & relevant)
                    # wrong-entity: the query names a person (PROBES); a fact about the OTHER person got into the
                    # top-5 (e.g. Eva's school recalled for "what scooter does Anton ride?").
                    persons = dict(data.PROBES[lang])
                    for name, _own in persons.items():
                        if name in query:
                            other = set().union(*(v for n, v in persons.items() if n != name))
                            wrong += len(found_keys & other)
                else:
                    negatives += 1
                    quiet += not injected
            failures += getattr(getattr(p, "_rescorer", None), "failures", 0)
            p.shutdown()
    finally:
        if not endpoint_url:
            rerank._OPENER = real_opener

    return {
        "mode": mode,
        "recall5": (hit5 / asked) if asked else 0.0,
        "hit5": hit5,
        "asked": asked,
        "wrong": wrong,
        "noise": (negatives - quiet) / negatives if negatives else 0.0,
        "quiet": quiet,
        "negatives": negatives,
        "p50_ms": _percentile(times, 0.50) * 1000.0,
        "p95_ms": _percentile(times, 0.95) * 1000.0,
        "calls": mock.requests if not endpoint_url else None,
        "failures": failures,
        "langs": langs,
    }


def table(rows: list[dict], endpoint: str | None) -> str:
    head = "| mode | recall@5 | hit5/asked | wrong-entity | noise | p50 (ms) | p95 (ms) |\n"
    head += "|---|---|---|---|---|---|---|\n"
    body = []
    for r in rows:
        recall = f"{r['recall5']:.0%}"
        body.append(f"| {r['mode']} | {recall} | {r['hit5']}/{r['asked']} | {r['wrong']} | "
                    f"{r['noise']:.0%} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} |")
    note = ("\nEndpoint: " + (endpoint or "mock (order only; latency is the mock's, not a server's)"))
    failed = sum(r.get("failures", 0) for r in rows)
    if failed:
        note += f"\nranked is None (fallbacks to coverage order): {failed}"
    return head + "\n".join(body) + note + "\n"


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Benchmark the rescoring modes over the multilingual set (RFC #9).")
    ap.add_argument("--langs", default="en,ru,zh", help="comma-separated language keys (default: en,ru,zh)")
    ap.add_argument("--modes", default=",".join(MODES), help="comma-separated modes to run (default: all five)")
    ap.add_argument("--endpoint", default=None,
                    help="a real /v1/rerank URL (llama-server); default: a mock ranker")
    ap.add_argument("--reverse", action="store_true", help="the mock ranker inverts the order (worst case)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a markdown table")
    ap.add_argument("--label", default=None, help="run label (model x device) stamped into every row")
    ap.add_argument("--out", default=None, help="append rows as JSON-lines to this file (accumulate runs)")
    args = ap.parse_args(argv)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    rows = [run(langs, mode, args.endpoint, args.reverse) for mode in modes]
    if args.label:
        for r in rows:
            r["label"] = args.label
    if args.out:
        # Overwrite: each cell writes its own file fresh, so re-runs never duplicate rows.
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(table(rows, args.endpoint))


if __name__ == "__main__":
    main(sys.argv[1:])