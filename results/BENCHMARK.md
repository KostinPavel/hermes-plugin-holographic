# Benchmark on the PR head

Every row of the raw files carries `git_sha` and `endpoint`; the runs below were made on head
commit `b6cbec5` with that commit's default config (`RECALL_MARGIN = 1.0` — the 0.6 proposal lives
in PR #17). Harness: `tools/provenance_probe.py`; raw rows:
`results/provenance_{multi,contrast_ru,contrast_en,contrast_zh}_{off,on}.jsonl`.

**Sets.** One multilingual set (18 languages — 360 real queries + 90 negatives) and the contrast sets
ru/en/zh (30 real + 45 negatives each): **450 real / 225 negatives per cell**.
**Modes.** `off` (no rerank), `reorder_only` (order only, window 0), `gate_only` (window 0.05 — the
shipped configuration; the pre-0.17.8 labels map as in README, "Benchmark" section).

| embeddings | mode | hit / 450 | wrong-entity | noise / 225 | output lines avg (max) |
|---|---|---|---|---|---|
| on | off | 433 | 28 | 73 | 1.80 (7) |
| on | reorder_only | 433 | 26 | 73 | 1.72 (5) |
| on | gate_only | 432 | 6 | 73 | 0.90 (5) |
| off | off | 430 (428 within the first five) | 25 | 73 | 1.80 (7) |
| off | reorder_only | 430 | 26 | 73 | 1.72 (5) |
| off | gate_only | 429 | 6 | 73 | 0.90 (5) |

- **hit** — the target fact is in the returned output; counting only the first five lines changes
  `off` in 2 queries (emb=off only), the longest output is 7 lines.
- **wrong-entity** — a real query whose output contains the partner fact of a confusable pair:
  27 of 28 (emb=on) are in the contrast sets, one in the multilingual set. Reordering moves the
  count by ±2 (the order changes what the coverage gate keeps); the window cut is what removes
  them — `gate_only` 28 → 6 (emb=on), 25 → 6 (emb=off).
- **noise** — negatives that returned anything. Decomposition (emb=off, `off` mode):
  `greeting_entity` 45/45 — a message that names a person recalls that person, by design per the
  PR thread; `topic_shared_words` 26/45; `vague` 2/45 (the upstream words-gate fix, PR #13,
  removed the stem collisions); the multilingual negatives 0/90.

**CPU latency** (rerank layer only, quality at 100%): `results/bench_p8_cpu.jsonl` —
bge-reranker-v2-m3 Q4_K_M, llama-server `--device none`, Ryzen 7 5800H (8 cores), pools avg 2–5
docs (max 16), 60 queries × 4 rerank modes = 240 calls, 0 fallbacks. Measured on `7f1efd6`, before
the upstream words-gate fix (which only shrinks the pools):

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
the pre-revert run (RECALL_MARGIN 0.6, `git_sha 7f1efd6`), the wider-store and admission sweeps and
the files removed by the PR housekeeping are in the same history.