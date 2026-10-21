# -*- coding: utf-8 -*-
"""Где теряется hard-цель: этап 1 (словесные кандидаты), обрезка top_k=40,
сама модель (порядок после реранка) или гейт/фильтры после него.

Логирует три точки конвейера prefetch для 30 реальных запросов:
search() -> candidates, Rescorer.scores() -> documents, итоговый recall.

    HERMES_AGENT_REPO=... python tools/loss_probe.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "contrast_off", Path(__file__).resolve().parent / "contrast_off.py")
co = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(co)

from hermes_plugin_holographic import rerank  # noqa: E402
from hermes_plugin_holographic.retrieval import FactRetriever  # noqa: E402

_search_log: list[tuple[str, list[int]]] = []          # (query, fact_id в порядке пула)
_scores_log: list[tuple[str, list[str], list[int]]] = []  # (query, documents, порядок индексов)

_orig_search = FactRetriever.search
_orig_scores = rerank.Rescorer.scores


def _search(self, query, *a, **kw):
    out = _orig_search(self, query, *a, **kw)
    _search_log.append((query, [f["fact_id"] for f in out]))
    return out


def _scores(self, query, documents):
    out = _orig_scores(self, query, documents)
    _scores_log.append((query, list(documents), [i for i, _ in out] if out else []))
    return out


FactRetriever.search = _search
rerank.Rescorer.scores = _scores

TARGET_CONTENT = {k: v[0] for k, v in co.contrast.FACTS.items()}


def _recall(block: str) -> list[int]:
    ids = []
    for line in (block or "").splitlines()[1:]:
        try:
            ids.append(int(line.split()[1][1:]))
        except (IndexError, ValueError):
            pass
    return ids


def run_mode(mode: str) -> None:
    import tempfile
    db = Path(tempfile.gettempdir()) / f"loss_probe_{mode}.db"
    for s in ("", "-shm", "-wal"):
        try:
            Path(str(db) + s).unlink()
        except OSError:
            pass
    spec = co.MODES[mode]
    config = {}
    if spec:
        config = {"rerank": "on", "rerank_mode": spec[0], "rerank_weight": spec[1],
                  "rerank_url": co.RERANK_URL, "rerank_model": "local-reranker"}
        if len(spec) > 2:
            config["rerank_margin"] = spec[2]
    p = co._provider(db, **config)
    back = co._fill(p)               # fid -> key
    key_of = {fid: key for fid, key in back.items()}
    content2key = TARGET_CONTENT

    print(f"\n=== режим {mode}: цель в пуле -> в documents -> после реранка -> в recall ===")
    print(f"{'цель':<18} {'канд':>7} {'docs':>7} {'ранг':>5} {'recall':>7}  диагноз / запрос")
    cases = [(pair, qi) for pair in co.contrast.PAIRS for qi in range(2)]
    for i, (pair, qi) in enumerate(cases):
        query, target = pair["queries"][qi]
        p.on_session_switch(f"loss_{i}", reset=True)
        block = p.prefetch(query)
        recall = _recall(block)
        fid_t = next((f for f, k in back.items() if k == target), None)

        cand_pos = doc_idx = rank = None
        n_cand = n_docs = 0
        for q, fids in _search_log:
            if q == query:
                n_cand = len(fids)
                if fid_t in fids:
                    cand_pos = fids.index(fid_t)
        for q, docs, order in _scores_log:
            if q == query:
                n_docs = len(docs)
                try:
                    doc_idx = docs.index(content2key[target])
                except ValueError:
                    doc_idx = None
                if doc_idx is not None and doc_idx in order:
                    rank = order.index(doc_idx) + 1
        _search_log.clear()
        _scores_log.clear()

        in_recall = recall.index(fid_t) + 1 if fid_t in recall else None
        if in_recall:
            verdict = "OK"
        elif cand_pos is None:
            verdict = "A: нет в словесном пуле"
        elif doc_idx is None:
            verdict = f"B: отсечён top_k (канд {n_cand} > {n_docs})"
        elif rank is None:
            verdict = "B?: индекс не вернулся"
        elif rank > 5:
            verdict = f"C: модель поставила глубже топ-5 (ранг {rank})"
        else:
            verdict = "D: ранг<=5, но вырезан гейт/фильтр"
        print(f"{target[:18]:<18} {str(cand_pos)+'/'+str(n_cand):>7} "
              f"{str(doc_idx)+'/'+str(n_docs):>7} {str(rank):>5} "
              f"{str(in_recall):>7}  {verdict} | {query[:40]}")
    p.shutdown()


if __name__ == "__main__":
    for m in ("reorder_0", "gate_0"):
        run_mode(m)