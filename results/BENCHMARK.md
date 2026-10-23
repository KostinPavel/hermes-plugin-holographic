# Benchmark on the PR head

Every row of the raw files carries `git_sha` and `endpoint`; the runs below were made on provider
commit `7f1efd6` with that commit's default config (`RECALL_MARGIN = 0.6` — the threshold change is
moving to its own PR). Harness: `tools/provenance_probe.py`; raw rows:
`results/provenance_{multi,contrast_ru,contrast_en,contrast_zh}_{off,on}.jsonl`.

**Sets.** One multilingual set (18 languages — 360 real queries + 90 negatives) and the contrast sets
ru/en/zh (30 real + 45 negatives each): **450 real / 225 negatives per cell**.
**Modes.** `off` (no rerank), `reorder_only` (order only, window 0), `gate_only` (window 0.05 — the
shipped configuration; the pre-0.17.8 labels map as in README, "Benchmark" section).

| embeddings | mode | hit / 450 | wrong-entity | noise / 225 | output lines avg (max) |
|---|---|---|---|---|---|
| on | off | 438 | 31 | 79 | 1.95 (7) |
| on | reorder_only | 438 | 32 | 79 | 1.86 (5) |
| on | gate_only | 437 | 6 | 79 | 0.97 (5) |
| off | off | 430 (428 within the first five) | 30 | 79 | 1.95 (7) |
| off | reorder_only | 430 | 32 | 79 | 1.84 (5) |
| off | gate_only | 429 | 6 | 79 | 0.96 (5) |

- **hit** — the target fact is in the returned output; `off` returns more than five lines in 45
  queries (max 7), and counting only the first five changes `off` in 2 queries (emb=off only).
- **wrong-entity** — a real query whose output contains the partner fact of a confusable pair; all
  such rows are in the contrast sets. The window cut (not the reordering) is what removes them:
  `reorder_only` == `off`, `gate_only` 31 → 6.
- **noise** — negatives that returned anything. 79/225 are word-sharing trap lines by construction
  (`greeting_entity` / `topic_shared_words`, agreed in the PR thread: a message that names a person
  is meant to recall that person).

**CPU latency** (rerank layer only, quality at 100%): `results/bench_p8_cpu.jsonl` —
bge-reranker-v2-m3 Q4_K_M, llama-server `--device none`, Ryzen 7 5800H (8 cores), pools avg 2–5
docs (max 16), 60 queries × 4 rerank modes = 240 calls, 0 fallbacks:

| mode | p50 ms | p95 ms |
|---|---|---|
| off | 3 | 6 |
| reorder_only | 94 | 222 |
| reorder_blend | 99 | 212 |
| gate_only | 101 | 229 |
| gate_blend | 97 | 214 |

**History.** The per-model table (bge/jina, CPU vs GPU, 360 queries, measured before the fixes) and
its raw rows (the `*_[...].jsonl` files) live at commit
[`634cfea`](https://github.com/itpartypattaya/hermes-plugin-holographic/blob/634cfea/results/BENCHMARK.md);
the wider-store and admission sweeps and their raw files are in the same history, together with the
files removed by the PR housekeeping.