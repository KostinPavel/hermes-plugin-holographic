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

**CPU latency** (rerank layer only, quality at 100%): `results/bench_cpu_t4.jsonl` —
bge-reranker-v2-m3 Q4_K_M, llama-server `--threads 4`, **Ryzen 7 2700 (this machine)**, 60 queries ×
4 rerank modes = 240 calls, 0 fallbacks, `git_sha 77c9325` (the head):

| mode | p50 ms | p95 ms |
|---|---|---|
| off | 3.7 | 9.2 |
| reorder_only | 178 | 411 |
| reorder_blend | 176 | 393 |
| gate_only | 180 | 398 |
| gate_blend | 184 | 413 |

The earlier CPU table (`results/bench_p8_cpu.jsonl`, `git_sha 7f1efd6`, kept for history) was measured
on a **different processor**: Ryzen 7 5800H laptop, msys2 llama-server `--device none`, before the
upstream words-gate fix — off p50 3 / p95 6 ms, rerank modes p50 94–101 / p95 212–229 ms, pools avg 2–5
docs (max 16). Same model and harness, but the two machines are not comparable; the 4-thread cap on the
new run is deliberate — it is closer to the small servers people run this on.

**CPU gate (2026-10-23, PR #10 point 3):** both rerankers measured in one run — llama-server
`--threads 4` on this machine (Ryzen 7 2700), both models **Q4_K_M**, protocol `--warmup 1
--repeats 5` (warmup passes excluded from every statistic, 620 measured calls per model),
`git_sha 19d7ced`. Raw rows: `results/bench_cpu_gate{,_calls}.jsonl` (bge),
`results/bench_cpu_gate_jina{,_calls}.jsonl` (jina):

| model (CPU, t4) | prefetch p50 ms | prefetch p95 ms | p95 < 300 ms |
|---|---|---|---|
| bge-reranker-v2-m3 Q4_K_M (`:1240`) | 343 | **448** | нет |
| jina-reranker-v2-base-multilingual Q4_K_M (`:1239`) | 121 | 152 | да |

Quality held in every mode on both: recall@5 100%, hit5 60/60, wrong-entity 0, noise 0%.
The 393–413 ms conflict noted above is confirmed at 448 ms p95 on the gate protocol; jina CPU
closes the 300 ms bar with margin and is the CPU candidate, bge stays the GPU default.

**Household dev sweep (2026-10-23, PR #10 point 5):** the fixed-label household set re-run on **dev only**
(test untouched — frozen until the maintainer's verdict on dev): `tools/provenance_probe.py --set household
--split dev`, seven modes, `--emb both`, GPU endpoints — bge Q4_K_M `:1235` (names-fit on, the default,
and `--names-fit off`) and jina Q4_K_M `:1236`. Per cell: 120 real queries + 90 negatives. Raw rows:
`results/provenance_household_dev{,_nfoff,_jina}_{off,on}.jsonl` (`git_sha ec3f941` for `dev_off`,
`0090424` for the rest).

bge, names-fit on — `named`/`confused`: real queries whose output contains such a fact; `lost`: per-target
(hit@5 hides partial multi-target losses); `neg`: negatives showing anything (90 per mode):

| mode | hit5 (emb off / on) | named | confused | lost (on / off) | neg (on / off) |
|---|---|---|---|---|---|
| off | 69 / 75 | 38 | 7 | 33 / 40 | 36 / 35 |
| reorder_only | 69 / 76 | 38 | 7 | 32 / 40 | 36 / 35 |
| gate_only (δ=0.05) | 69 / 76 | 9 | 3 | 37 / 45 | 36 / 35 |
| gate_min2 (a: `rerank_window_min: 2`) | 69 / 76 | 34 | 4 | 32 / 40 | 36 / 35 |
| gate_named (b: `rerank_window_named: on`) | 69 / 76 | 38 | 3 | 37 / 45 | 36 / 35 |
| win03 (δ=0.03) | 69 / 76 | 8 | 2 | 39 / 47 | 36 / 35 |
| win10 (δ=0.10) | 69 / 76 | 9 | 3 | 36 / 44 | 36 / 35 |

- names-fit off, emb=on: off hit5 78 → 79 in every mode (the 10:34 numbers reproduce), confused 7 → 3–4.
- jina, the same cells: hit5 identical (69 / 75 → 76); confused cleans harder — `gate_only` 7 → 1,
  `win03` 7 → 0 (gate_named 7 → 4), nothing added in any mode; `lost` emb=on: 33 → 32 (`gate_min2`),
  40 (`gate_only`), 41 (`win03`).
- Exact two-sided McNemar on the hit5 pairs vs `off` (`tools/mcnemar_gate.py`): the best case anywhere is
  1 rescued / 0 lost, p = 1.0 — the ≥ +5 points at p < 0.05 branch of the gate is out of reach on dev.
  The other branch holds for every gate/window mode: **0 right facts lost in any mode**, confused
  7 → 2–4 (bge) / 7 → 0–4 (jina; `gate_only` 1, `win03` 0), negatives constant across modes.
- The window cut is what removes the confused facts (`reorder_only` changes nothing); `gate_min2` —
  variant (a) — is the only mode holding per-target losses at the `off` level (32–40 vs 33–40).

**History.** The per-model table (bge/jina, CPU vs GPU, 360 queries, measured before the fixes) and
its raw rows (the `*_[...].jsonl` files) live at commit
[`634cfea`](https://github.com/itpartypattaya/hermes-plugin-holographic/blob/634cfea/results/BENCHMARK.md);
the pre-revert run (RECALL_MARGIN 0.6, `git_sha 7f1efd6`), the wider-store and admission sweeps and
the files removed by the PR housekeeping are in the same history.