# -*- coding: utf-8 -*-
"""Provenance probe: откуда каждый факт в top-5 перед ходом (лексический / смысл / имена),
что фильтр coverage вырезал и почему цель не найдена (классы A/D/C).

Патчит только экземпляр провайдера в тестовом прогоне (продакшн-код не меняется):
  search      -> пул (кандидаты после blend)
  relevant    -> kept / blocked (+ позиция в пуле, top-5 пула = класс D)
  by_meaning  -> источник 'meaning'
  entity_facts-> источник 'entity'
  rescorer.scores -> порядок реранка (top-5 реранка)
  _not_recent / _recallable -> 'recent' / 'unsafe'

Запуск:
    PYTHONPATH=~/.hermes/hermes-agent python -I tools/provenance_probe.py \
        --set multi --langs en,ru,zh,de,ja,it,nl,th --modes off,reorder_only,gate_only
    PYTHONPATH=~/.hermes/hermes-agent python -I tools/provenance_probe.py \
        --set contrast_ru --modes off,gate_only --emb on
    # --endpoint http://127.0.0.1:1235/v1/rerank   реальный реранк (иначе мок)
    # --emb on|off|both                              embeddings с :1237 (по умолчанию off)

Строки: results/provenance_<set>_<emb>_<mode>.jsonl (по строке на запрос).
"""
from __future__ import annotations

import argparse
import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "hermes_plugin_holographic", _REPO / "__init__.py", submodule_search_locations=[str(_REPO)])
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hermes_plugin_holographic"] = _mod
_spec.loader.exec_module(_mod)

from hermes_plugin_holographic import HolographicMemoryProvider  # noqa: E402
from hermes_plugin_holographic import _sendable as _holographic_sendable  # noqa: E402
from hermes_plugin_holographic import embeddings as _holographic_embeddings  # noqa: E402

# mode -> (rerank_mode, weight[, rerank_margin[, criterion]]) — та же номенклатура, что benchmark_rerank
MODES = {
    "off": None,
    "reorder_only": ("reorder_before_gate", "0.0"),
    "reorder_blend": ("reorder_before_gate", "0.5"),
    "gate_only": ("gate_reads_score", "0.0"),
    "gate_blend": ("gate_reads_score", "0.5"),
    # свип порога (блок 2): все ветки gate_reads_score
    "gate_m12": ("gate_reads_score", "0.0", "1.2"),
    "gate_m11": ("gate_reads_score", "0.0", "1.1"),
    "gate_m105": ("gate_reads_score", "0.0", "1.05"),
    "gate_m10": ("gate_reads_score", "0.0", "1.0"),
    # верхний критерий (всё-или-ничего по пулу): gap/пол/пол+гап вместо ratio-вето
    "topgap": ("gate_reads_score", "0.0", "1.5", "topgap:0.05"),
    "gate_gap03": ("gate_reads_score", "0.0", "1.0", "topgap:0.03"),
    "gate_gap02": ("gate_reads_score", "0.0", "1.0", "topgap:0.02"),
    "topfloor": ("gate_reads_score", "0.0", "1.5", "topfloor:0.5"),
    "topfg": ("gate_reads_score", "0.0", "1.5", "topfloorgap:0.5:0.05"),
    # нижний критерий (побитовое отсечение хвоста), вето выключено (margin 1.0)
    "botwin10": ("gate_reads_score", "0.0", "1.0", "botwin:0.10"),
    "botwin05": ("gate_reads_score", "0.0", "1.0", "botwin:0.05"),
    "botfloor5": ("gate_reads_score", "0.0", "1.0", "botfloor:0.5"),
    "botfloor3": ("gate_reads_score", "0.0", "1.0", "botfloor:0.3"),
}

import hermes_plugin_holographic as _H  # noqa: E402  (hook-и гейта для подмены критерия)

_ORIG_CLEAR = _H._margin_clears
_ORIG_CUT = _H._bottom_cut


def apply_criterion(crit: str | None) -> None:
    """Критерий гейта для режима: 'topgap:δ' | 'topfloor:τ' | 'topfloorgap:τ:δ' |
    'botwin:δ' | 'botfloor:τ'; None = оригинал (_margin_clears / no-cut)."""
    _H._margin_clears = _ORIG_CLEAR
    _H._bottom_cut = _ORIG_CUT
    if not crit:
        return
    parts = crit.split(":")
    k = parts[0]
    if k == "topgap":          # вето проходит, когда отрыв топа ≥ δ (вместо ratio)
        d = float(parts[1])
        _H._margin_clears = lambda b, s, m: s is None or (b - s) >= d
    elif k == "topfloor":      # вето проходит, когда сам топ ≥ τ
        t = float(parts[1])
        _H._margin_clears = lambda b, s, m: b >= t
    elif k == "topfloorgap":   # порог с гапом: оба условия
        t, d = float(parts[1]), float(parts[2])
        _H._margin_clears = lambda b, s, m: b >= t and (s is None or (b - s) >= d)
    elif k == "botwin":        # оставить всё в пределах δ от лучшего (хвост снизу)
        d = float(parts[1])
        _H._bottom_cut = lambda scores: [i for i, s in enumerate(scores) if scores[0] - s <= d]
    elif k == "botfloor":      # оставить всё с отдельным скором ≥ τ
        t = float(parts[1])
        _H._bottom_cut = lambda scores: [i for i, s in enumerate(scores) if s >= t]
    else:
        raise ValueError(f"unknown criterion: {crit}")

EMB_URL = os.environ.get("EMB_URL", "http://127.0.0.1:1237/v1")
EMB_MODEL = os.environ.get("EMB_MODEL", "bge-m3")
ENDPOINT = os.environ.get("RERANK_URL", "http://127.0.0.1:1235/v1/rerank")


# ---------------------------------------------------------------- данные наборов

def load_multilingual():
    spec = importlib.util.spec_from_file_location(
        "holo_lang_data", _REPO / "tests" / "test_holographic_languages.py")
    assert spec and spec.loader
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    return data


def load_contrast(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "tools" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- провайдер

def make_provider(db: Path, mode: str, emb, extra: dict | None = None) -> HolographicMemoryProvider:
    spec = MODES[mode]
    config = {"db_path": str(db), "hrr_dim": 1024}
    if spec:
        config.update({"rerank": "on", "rerank_mode": spec[0], "rerank_weight": spec[1],
                       "rerank_url": ENDPOINT, "rerank_model": "probe-reranker"})
        if len(spec) > 2:
            config["rerank_margin"] = spec[2]
        apply_criterion(spec[3] if len(spec) > 3 else None)
    else:
        apply_criterion(None)
    if emb == "on":
        config.update({"embeddings": "on", "embedding_url": EMB_URL, "embedding_model": EMB_MODEL})
    if extra:
        config.update(extra)
    p = HolographicMemoryProvider(config=config)
    p.initialize(session_id="prov")
    return p


class Probe:
    """Собирает provenance одного prefetch. Контекст сбрасывается на каждый запрос."""

    def __init__(self, p: HolographicMemoryProvider):
        self.p = p
        self.ctx: dict = {}
        self._orig = {}
        r = p._retriever
        self._wrap(r, "search", self._search)
        self._wrap(r, "relevant", self._relevant)
        self._wrap(r, "by_meaning", self._by_meaning)
        self._wrap(r, "entity_facts", self._entity_facts)
        self._wrap(p, "_not_recent", self._not_recent)
        self._wrap(p, "_recallable", self._recallable)
        if p._rescorer is not None:
            self._wrap(p._rescorer, "scores", self._scores)

    def _wrap(self, obj, name: str, wrapper):
        orig = getattr(obj, name, None)
        if orig is None:
            return
        self._orig[(id(obj), name)] = orig
        setattr(obj, name, wrapper(orig))

    # -- вызовы пайплайна ------------------------------------------------
    def _search(self, orig):
        def call(*a, **kw):
            out = orig(*a, **kw)
            self.ctx["pool"] = [f["fact_id"] for f in out]
            self.ctx["order"] = list(self.ctx["pool"])
            return out
        return call

    def _scores(self, orig):
        def call(query, docs):
            ranked = orig(query, docs)
            if ranked is not None:
                pool = self.ctx.get("pool") or []
                self.ctx["rerank_top5"] = [pool[i] for i, _ in ranked[:5] if i < len(pool)]
                self.ctx["ranked"] = ranked is not None
            else:
                self.ctx["ranked"] = False
            return ranked
        return call

    def _relevant(self, orig):
        def call(query, candidates, min_coverage=0.3):
            cids = [f["fact_id"] for f in candidates]
            self.ctx["gate_called"] = True
            self.ctx["gate_order"] = list(cids)
            kept = orig(query, candidates, min_coverage)
            k = {f["fact_id"] for f in kept}
            for pos, fid in enumerate(cids):
                if fid in k:
                    self.ctx.setdefault("tags", {}).setdefault(fid, set()).add("words")
                else:
                    self.ctx.setdefault("blocked", {})[fid] = pos
            return kept
        return call

    def _by_meaning(self, orig):
        def call(*a, **kw):
            out = orig(*a, **kw)
            for f in out:
                self.ctx.setdefault("tags", {}).setdefault(f["fact_id"], set()).add("meaning")
            return out
        return call

    def _entity_facts(self, orig):
        def call(*a, **kw):
            out = orig(*a, **kw)
            for f in out:
                self.ctx.setdefault("tags", {}).setdefault(f["fact_id"], set()).add("entity")
            return out
        return call

    def _not_recent(self, orig):
        def call(key, fact, turn):
            ok = orig(key, fact, turn)
            if not ok:
                self.ctx.setdefault("tags", {}).setdefault(fact["fact_id"], set()).add("recent")
            return ok
        return call

    def _recallable(self, orig):
        def call(fact):
            ok = orig(fact)
            if not ok:
                self.ctx.setdefault("tags", {}).setdefault(fact["fact_id"], set()).add("unsafe")
            return ok
        return call

    # -- обёртка запроса --------------------------------------------------
    def prefetch(self, query: str) -> tuple[str, dict]:
        self.ctx = {"tags": {}, "blocked": {}, "gate_called": False,
                    "pool": [], "rerank_top5": [], "ranked": None}
        block = self.p.prefetch(query)
        ctx = dict(self.ctx)
        ctx["tags"] = {k: sorted(v) for k, v in ctx["tags"].items()}
        ctx["blocked"] = dict(ctx["blocked"])
        return block, ctx


def catch_up_vectors(p: HolographicMemoryProvider) -> bool:
    """Векторы фактов делаются только в _catch_up_vectors (after-turn хук ядра).
    В прогоне драйвера его никто не вызывает -> индекс пуст -> meaning=None.
    Гоняем вручную, пока есть pending: ready() проверяет «хотя бы один вектор»
    (LIMIT 1), а бюджет одного _catch_up_vectors (embedding_catch_up_seconds)
    отсекает хвост — на wide-базе (600+ фактов) это вырезало цели real-запросов
    (fact_id 1..25): «готово» при 25 неэбедженных."""
    if not p._embedding:
        return True
    idx = p._retriever.vector_index(p._embedding.tag)
    for _ in range(60):
        try:
            pending = idx.pending(_holographic_embeddings.BATCH, allowed=_holographic_sendable)
        except Exception:
            pending = None
        if pending is None:  # API неожиданного вида — старая эвристика
            if idx.ready():
                return True
        elif not pending:
            return True
        p._catch_up_vectors()
    return True


def parse_ids(block: str) -> list[int]:
    ids = []
    if not block:
        return ids
    for line in block.splitlines()[1:]:
        try:
            ids.append(int(line.split()[1][1:]))
        except (IndexError, ValueError):
            pass
    return ids


def classify(fid: int | None, ctx: dict) -> str:
    """Почему цель не в recall (для real-запросов)."""
    if fid is None:
        return "-"
    if fid in ctx["tags"] and any(t in ctx["tags"][fid] for t in ("recent", "unsafe")):
        return "dropped:" + ",".join(ctx["tags"][fid])
    pool = ctx.get("pool") or []
    if fid not in pool:
        return "A: нет в словесном пуле"
    blocked = ctx.get("blocked") or {}
    if fid in blocked:
        pos = blocked[fid]
        r5 = fid in (ctx.get("rerank_top5") or [])
        return "D: вырезан coverage" + (" (top-5 реранка!)" if r5 else f" (пул#{pos})")
    if not ctx.get("gate_called"):
        # gate_reads_score адмитит пул без coverage: цель либо ниже top-5, либо отвергнута margin
        order = ctx.get("gate_order") or pool
        if fid in order[:5]:
            return "gate: отвергнут margin (в top-5 порядка!)"
        return "C: ниже top-5"
    # coverage пустил, но не в recall: свежесть/лимит top-5
    order = ctx.get("gate_order") or pool
    pos = order.index(fid) if fid in order else 999
    return f"C: ниже top-5 (пул#{pos})"


# ---------------------------------------------------------------- прогоны

def run_multilingual(langs, modes, emb, endpoint, extra: dict | None = None) -> list[dict]:
    data = load_multilingual()
    rows = []
    for mode in modes:
        for lang in langs:
            db = Path(tempfile.gettempdir()) / f"prov_ml_{lang}_{mode}.db"
            for s in ("", "-shm", "-wal"):
                try:
                    Path(str(db) + s).unlink()
                except OSError:
                    pass
            p = make_provider(db, mode, emb, extra)
            back = {}
            for key, (text, entities) in data.CORPUS[lang].items():
                r = json.loads(p.handle_tool_call(
                    "fact_store", {"action": "add", "content": text, "entities": entities}))
                back[int(r["fact_id"])] = key
            probe = Probe(p)
            if emb == "on":
                ok = catch_up_vectors(p)
                print(f"  [{lang}] векторы готовы: {ok}")
            persons = dict(data.PROBES[lang])
            for i, (query, relevant) in enumerate(data.QUERIES[lang]):
                p.on_session_switch(f"pv{li(mode, lang)}_{i}", reset=True)
                t0 = time.perf_counter()
                block, ctx = probe.prefetch(query)
                ms = (time.perf_counter() - t0) * 1000
                found = parse_ids(block)
                keys = [back[f] for f in found if f in back]
                kind = "real" if relevant else "chit"
                hit = bool(set(keys) & set(relevant)) if relevant else None
                wrong_keys = []
                if relevant:
                    for name in persons:
                        if name in query:
                            other = set().union(*(v for n, v in persons.items() if n != name))
                            wrong_keys = sorted(set(keys) & set(other))
                miss_cls = "-"
                if relevant and not hit:
                    # цель(и) без hit: класс по первой ненайденной
                    miss_cls = next((classify(next((f for f, k in back.items() if k == rk), None), ctx)
                                     for rk in relevant if rk not in keys), "-")
                rows.append({"set": "multi", "lang": lang, "mode": mode, "emb": emb,
                             "kind": kind, "q": query, "keys": keys,
                             "sources": {str(f): ctx["tags"].get(f, ["pool-only"])
                                         for f in found},
                             "hit": hit, "wrong": wrong_keys,
                             "miss_cls": miss_cls, "ms": round(ms, 1),
                             "pool_n": len(ctx.get("pool") or []),
                             "gate_called": ctx.get("gate_called"),
                             "rerank_top5": ctx.get("rerank_top5") or []})
            p.shutdown()
    return rows


def li(mode, lang):  # префикс сессии — уникален на режим+язык
    return f"{mode}_{lang}"


def run_contrast(set_name, modes, emb, endpoint, extra: dict | None = None) -> list[dict]:
    contrast = load_contrast(set_name)
    rows = []
    for mode in modes:
        db = Path(tempfile.gettempdir()) / f"prov_{set_name}_{mode}.db"
        for s in ("", "-shm", "-wal"):
            try:
                Path(str(db) + s).unlink()
            except OSError:
                pass
        p = make_provider(db, mode, emb, extra)
        back = {}
        for key, (text, entities) in contrast.FACTS.items():
            r = json.loads(p.handle_tool_call(
                "fact_store", {"action": "add", "content": text, "entities": entities}))
            back[int(r["fact_id"])] = key
        probe = Probe(p)
        if emb == "on":
            ok = catch_up_vectors(p)
            print(f"  [{set_name}] векторы готовы: {ok}")
        for pi, pair in enumerate(contrast.PAIRS):
            partner = [pair["queries"][1 - i][1] for i in range(len(pair["queries"]))]
            for i, (query, target) in enumerate(pair["queries"]):
                p.on_session_switch(f"cp{pi}_{i}", reset=True)
                block, ctx = probe.prefetch(query)
                found = parse_ids(block)
                keys = [back[f] for f in found if f in back]
                own_pos = keys.index(target) + 1 if target in keys else None
                part = partner[i]
                part_pos = None if part == target else (keys.index(part) + 1 if part in keys else None)
                miss_cls = classify(next((f for f, k in back.items() if k == target), None), ctx) \
                    if own_pos is None else "-"
                rows.append({"set": set_name, "lang": set_name, "mode": mode, "emb": emb,
                             "kind": "real", "q": query, "keys": keys, "target": target,
                             "own_pos": own_pos, "partner_pos": part_pos,
                             "sources": {str(f): ctx["tags"].get(f, ["pool-only"]) for f in found},
                             "hit": own_pos is not None, "wrong": part_pos is not None and part_pos <= 5,
                             "miss_cls": miss_cls, "pool_n": len(ctx.get("pool") or []),
                             "gate_called": ctx.get("gate_called"),
                             "rerank_top5": ctx.get("rerank_top5") or []})
        for i, (query, cat) in enumerate(contrast.CHITCHAT):
            p.on_session_switch(f"cq{i}", reset=True)
            block, ctx = probe.prefetch(query)
            found = parse_ids(block)
            keys = [back[f] for f in found if f in back]
            rows.append({"set": set_name, "lang": set_name, "mode": mode, "emb": emb,
                         "kind": "chit", "cat": cat, "q": query, "keys": keys,
                         "sources": {str(f): ctx["tags"].get(f, ["pool-only"]) for f in found},
                         "hit": None, "wrong": [], "miss_cls": "-",
                         "pool_n": len(ctx.get("pool") or []),
                         "gate_called": ctx.get("gate_called"),
                         "rerank_top5": ctx.get("rerank_top5") or []})
        p.shutdown()
    return rows


# ---------------------------------------------------------------- отчёт

def summarize(rows: list[dict]) -> None:
    modes = sorted({(r["mode"], r["emb"]) for r in rows})
    print(f"\n{'режим':<14} {'emb':<4} {'real':>5} {'hit':>5} {'шум':>5} "
          f"{'wrong5':>6} | источники recall (доминирует) | классы промахов")
    for mode, emb in modes:
        sel = [r for r in rows if r["mode"] == mode and r["emb"] == emb]
        real = [r for r in sel if r["kind"] == "real"]
        chit = [r for r in sel if r["kind"] == "chit"]
        hits = sum(bool(r["hit"]) for r in real)
        noise = sum(bool(r["keys"]) for r in chit)
        wrong = sum(1 for r in real if r["wrong"])
        # источники всех recalled-фактов
        src: dict[str, int] = {}
        for r in sel:
            for tags in r["sources"].values():
                key = "+".join(t if t != "pool-only" else "words?" for t in tags) \
                    if "pool-only" not in tags else "words?(без coverage — gate)"
                src[key] = src.get(key, 0) + 1
        src_s = ", ".join(f"{k}:{v}" for k, v in sorted(src.items(), key=lambda x: -x[1]))
        classes: dict[str, int] = {}
        for r in real:
            if r["kind"] == "real" and not r["hit"]:
                classes[r["miss_cls"]] = classes.get(r["miss_cls"], 0) + 1
        cls_s = "; ".join(f"{k}:{v}" for k, v in sorted(classes.items(), key=lambda x: -x[1])) or "—"
        print(f"{mode:<14} {emb:<4} {len(real):>5} {hits:>5} {noise:>5}/{len(chit):<3} "
              f"{wrong:>6} | {src_s} | {cls_s}")


def main(argv):
    global ENDPOINT
    ap = argparse.ArgumentParser(description="Provenance probe: источники фактов в prefetch")
    ap.add_argument("--set", default="multi", help="multi | contrast_ru | contrast_en | contrast_zh")
    ap.add_argument("--langs", default="en,ru,zh,de,ja,it,nl,th",
                    help="языки для --set multi (по умолчанию: ключевые из бенча)")
    ap.add_argument("--modes", default="off,reorder_only,gate_only")
    ap.add_argument("--emb", default="off", choices=["off", "on", "both"])
    ap.add_argument("--endpoint", default=ENDPOINT, help="/v1/rerank URL (мок, если не задан)")
    ap.add_argument("--out", default=None, help="каталог JSONL (по умолчанию results/)")
    args = ap.parse_args(argv)

    ENDPOINT = args.endpoint

    out_dir = Path(args.out) if args.out else (_REPO / "results")
    out_dir.mkdir(exist_ok=True)
    embs = ["off", "on"] if args.emb == "both" else [args.emb]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_rows: list[dict] = []

    # мок-раннер, если endpoint не задан — как в benchmark_rerank
    if not args.endpoint:
        print("внимание: endpoint не задан — используется мок-порядок (без реальных скоров)")

    for emb in embs:
        if args.set == "multi":
            rows = run_multilingual([lg.strip() for lg in args.langs.split(",") if lg.strip()],
                                    modes, emb, args.endpoint)
        else:
            rows = run_contrast(args.set, modes, emb, args.endpoint)
        path = out_dir / f"provenance_{args.set}_{emb}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n=== {args.set} emb={emb}: {len(rows)} запросов -> {path}")
        summarize(rows)
        all_rows.extend(rows)
    return all_rows


if __name__ == "__main__":
    main(sys.argv[1:])