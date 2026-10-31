# The presence gate: design note (proposed for 0.26.0)

Working note for the presence-gate PR, mapped line by line to the acceptance conditions in issue #59
(conditions 0–8 of comment 6523787794). Status: design agreed in the thread; Blocks I–II are implemented on
this branch (`1ec319c`: config, state, generations, snapshot ingestion; `c6c6a29`: the gate in `_not_recent` and
the grace window), the Block III tests land with them. The gate stays off by default.
The gate is an experiment behind `prefetch_presence_gate: false` — off is byte-identical to today (condition 6).

## Why

The reset that guards the 6-turn window is coarse: compaction keeps a protected tail, and the core stamps the
prefetch block into the user row's `api_content` and replays it while the row is in context. After
`on_pre_compress` clears `_injected`, a fact the model still sees can be injected a second time; the same happens
when compaction fails after the hook ran. A presence check fixes exactly this; the reset stays as the fallback.
Baseline on this store (see Metrics): duplicates exist and are countable; misses are not — the gate only claims
to suppress repeats of injections the provider tracks.

## What the gate does

A fact is suppressed only if **this key** injected it (`_injected`, keyed by `_revision(fact)`) **and** it is
either visible in the current-generation snapshot or was injected after that snapshot was taken (grace: that
injection's FIRST expiry decision only - one turn, never more).
Everything else follows today's rules: not-injected facts, other stores' `#N`, foreign blocks, prose — nothing
changes for them.

## State

- `_injected[key][revision]` stores, per injection event, **the exact line the plugin emitted** — cut, collapsed
  and suffixed as built — plus the turn. A list per revision, because `(by meaning)` depends on the retrieval
  path of that event: the same revision can be emitted with and without the suffix, and each event must match
  its own row.
- A **compaction generation** counter advanced by `on_pre_compress`, plus a **state epoch** advanced whenever
  tracked state is cleared (reset/rewind). With the gate on the hook does not clear `_injected`; it drops the
  snapshot and advances the generation (condition 0).
- `_snapshots[key] = {generation, epoch, turn, lines, pairs, summary}` filled by `sync_turn`: key, generation,
  epoch and turn are captured **under the lock before the view is copied**, the copy is parsed **outside** the
  lock, and the publish under the lock rejects the snapshot if any captured value moved - a compaction landing
  mid-parse fails the check instead of publishing the old view as current (the maintainer's implementation
  note). Past a compaction (`generation > 0`) the publish also requires a summary row this key has not been
  credited with yet: the summary's **signature** (sha1 of the row text) must differ from the last one credited -
  the previous compaction's row, or a paste of it, proves nothing. A held view is never replaced by an older
  turn's. And a turn arriving with `messages=None` or `messages=[]` now **drops** the held snapshot outright:
  empty turns prove nothing. Every rejection leaves the gate on today's window.

## Two zones, two representations — no cross-form search

| Zone | What the context physically holds | Artifact stored per event | Match |
|---|---|---|---|
| Recall block (`<memory-context>`) | **only the cut form**: collapsed, ≤300-char cap, suffixes | the emitted line, verbatim | whole line ↔ whole line inside a `<memory-context>` block |
| `fact_store` JSON results | **only the full form**: `content`, uncut | `(fact_id, content)` of the revision | `json.loads` the result, compare the pair |

Neither form ever appears in the other zone, so nothing is searched in both versions; the gate never
reconstructs a representation it did not produce itself.

Provenance decides whether a row is read at all: a recall line counts only inside a `<memory-context>` block in
a **user row's `api_content`** that carries the core's system note (a pasted block - in `api_content` or in
`content` - carries none), and a pair counts only from a `tool` row whose `tool_name` is exactly `fact_store`.
Anything unattributable (no `tool_name`, malformed JSON, an unknown block) is skipped: the gate fails open to
today's window.

- **`export` is out of scope**: its markdown carries a 500-char cut plus entity and source suffixes, so
  `(fact_id, content)` cannot be compared there. Documented, not matched (a cut→cut comparison would only
  reintroduce the ambiguity the design avoids).
- **`history` works in our favour**: the result carries the current revision (equal → suppress) and older ones
  (unequal → no effect).
- **The budget needs nothing**: `_within_budget` drops lines whole and `_remember` runs after it, so a dropped
  fact is never recorded (verified in code order).

## Escaping — three layers, each neutralised once

| Layer | Where it lives | How the design handles it |
|---|---|---|
| JSON escapes: `\"`, `\\`, `\n`, control chars (mandatory) and `\uXXXX` (only pre-#62, `ensure_ascii=True`) | tool results; old messages keep the `\u` form | `json.loads` undoes all of it — pre-#62 and post-#62 messages parse identically, so **no migration** and no dependency on PR #62 |
| Render: collapse + 300-cap + suffixes | block only | storing the as-emitted line — the form is never recomputed |
| Nothing (prose is plain text) | quotes, paraphrases, foreign blocks | out of scope by the maintainer's decision |

Consequence: the gate never needs to know which characters are escaped anywhere.

## The cut: 299

The cap is `_PREFETCH_LINE_CHARS = 300`; a longer line becomes `flat[:limit - 1] + "…"` — 299 characters and an
ellipsis. The 299 is derived (`limit - 1`), not stored: no second constant is introduced, so the cap cannot
drift from its cut. Presence code and test arithmetic derive the boundary from `_PREFETCH_LINE_CHARS` /
`_one_line`; exactly one pinned test asserts `299 + "…"` so a silent cap change fails CI.

**Cut-line policy — option (a)** (the maintainer's pick-one): compare the stored cut line against the cut form
of the revision in question; two revisions differing only past the cap share one line, which yields a rare false
"visible" — accepted, documented and tested. Option (b) (a cut line proves nothing) is rejected because every
fact longer than 299 characters would then never suppress, i.e. the very duplicates this gate exists to stop.

## Spill policy

An injection whose block was spilled **keeps today's window** instead of being released by absence: the core's
opt-in spill (`tools/hook_output_spill.py`, as the maintainer named it) replaces oversized replayed text with
`preview_head`/`preview_tail` plus a path, so a fresh snapshot may show the preview instead of the block;
treating that as "not visible" would hide and re-show the facts on alternating turns. For recall specifically the
call sits in `memory_manager.py` (`spill_if_oversized`, "Opt-in spill limits recall replayed with the user turn's
api_content"): the base cap (default 10 000) applies only when the core's `memory.prefetch_spill_enabled` is
true — otherwise it is raised to ×10 / ≥100 000. Order verified: `_remember` runs inside `prefetch`, the core spills after it returns. Test: the gate's two
verdicts equal the gate-off verdicts across a spilled preview - a fixed preview; a turn-by-turn flip-flop
simulation is not built (say so rather than promise it).

## Gate placement

Inside `_not_recent` — the single choke point all four call sites share (rerank branch, `_off_layout`, meaning,
entity). With nothing visible the behaviour is exactly today's window, top-5 included (condition 5).

## Documented limits

- Not tracked: injections lost to a restart, to the `_injected` caps, to a session switch; facts first seen only
  through a tool call were never injected.
- The first turn after a compaction can still duplicate a tail fact: `on_pre_compress` sees the list before
  compaction and the provider cannot tell which rows survived.
- A fact whose text contains the literal fence (`<memory-context>`) may be trimmed by the core's
  `sanitize_context`; its stored line then fails to appear — a rare edge whose only effect is a missed
  suppression.
- System-prompt copies are **not** a visibility zone - although facts from MEMORY.md can be there and in the
  plugin's store at the same time: MEMORY.md / USER.md sit in the system prompt in full, so such a fact can be
  re-injected once its block leaves. Prose stays out of scope by the maintainer's decision; whether a third
  zone is wanted belongs in #59, not in this PR.
- A forged, never-before-seen `[CONTEXT COMPACTION ...]` text typed inside a compaction window could still pass
  as that compaction's summary (no database flag arrives with a live message); a paste of an already-credited
  row cannot.

## Tests (condition 7, plus design-specific ones)

Built the way the core builds context (`build_memory_context_block` wrapped into `api_content`):

1. compacted head, kept tail — the tail's fact is not re-injected, the head's one is; its off-path branch runs
   main's logic and pins the duplicate (`out[False] == (True, True)`); the file itself needs this branch's
   `_remember(..., line)` signature, so it does not run on `main` as-is (stated in the PR);
2. another provider's block and a user message containing `#5 [` — no effect;
3. two `user`-scope stores with the same fact id — independent;
4. a revised fact — old text visible does not suppress the new revision (`test_a_reworded_fact_is_recalled_again`
   stays meaningful);
5. prefetch before the snapshot arrives — grace window, not silence;
6. spilled block — the policy above, no flip-flop;
7. a line emitted with `(by meaning)` matches only its own row, never the suffix-less one;
8. a legacy `\uXXXX` tool result still counts as visible (pre-#62 messages);
9. `export` output does not count as visibility;
10. `messages=None` / `messages=[]` — the held snapshot is dropped (fail open to the window), pinned; a
    snapshot whose generation moved during its parse never publishes;
11. only a NEW summary row proves the new compaction: the already-credited one keeps the publish rejected;
12. a pasted block (no core note, either field), malformed fact_store JSON and an id-only match are never
    visibility; the same id with other content stays a different fact;
13. grace holds the injection's first expiry decision only, then today's window decides again;
14. fully suppressed: the off path returns the block (the candidate is retrievable), the on path returns `""`.

## Metrics

Definitions: a duplicate = a `(session, fact_id)` key whose fact shows in ≥2 rows live now (`active = 1`),
lines read structurally (`<memory-context>`, our `- #N [trust]` shape only). Snapshot of this store, 2026-10-30:
**115 keys in 11 sessions** (101 in 10 the same morning — the store grew); of them **5 row-byte-identical twins
(kept in, per the maintainer)**, **106 identical line text**, **9 differing text under one `#N`** (a revision or
two stores sharing an id — the replayed row does not name its store, so pairs are grouped by text and the 9 stay
a labeled residual); cause between first and last live row: **8 compaction-preceded, 107 other** — window expiry
in uncompacted context dominates ("preceded by", not "caused by"). This is a structural proxy — rows live at
once, not observed re-injections — and the cause split is an operational signature (a summary row between the
live rows), not causation. The **after** number needs a live run with the gate on: the counting script and the
protocol ship with the PR, and the numbers follow from that run on one store — the maintainer's decision input
(condition 8).

## What this PR measures: the gate's own cost, on synthetics

The project records no baseline for `sync_turn` or prefetch latency (its timing tests pin the screen), so
`tools/presence_delta.py` measures the baseline in-process: every scenario runs gate off and on — same provider,
same inputs, phases interleaved to cancel drift — and reports the added cost next to each phase. Inputs are
synthetic and neutral (blocks via `build_memory_context_block`, `fact_store` JSON in both encodings, noise rows
without `<memory-context>`, one 200 KB tool row); the output is aggregates only.

| scenario | off p50 | on p50 | added p50 / p95 |
|---|---|---|---|
| sync, 10 rows | 0.001 ms | 0.115 ms | **+0.11 / 0.13 ms** |
| sync, 100 rows | 0.002 ms | 0.282 ms | **+0.28 / 0.29 ms** |
| sync, 1000 rows | 0.002 ms | 1.929 ms | **+1.93 / 1.95 ms** (~2 µs/row) |
| prefetch, no snapshot | 1.377 ms | 1.371 ms | ~0 (within noise) |
| prefetch, snapshot present | 1.366 ms | 1.368 ms | ~0 (within noise) |

Byte-identity without a snapshot: **True** (off == on per call). State per tracked injection: ~55 bytes of
emitted-line text **on these sample lines** (real lines run longer, so this is a sample figure, not a bound;
caps: 512 revisions x 8 emitted lines per revision, 16 sessions). Against the core's budgets — a 5 s
sync drain, a 30 s background stall — 1000 rows add ≈0.04% of the drain bound. Two runs (30/40 pairs) reproduced
every figure within ±0.5%. One machine, fixed inputs: a cost, reproducible in any CI — **not** a live-session
p95.

## What this PR does not claim

- **No after-numbers**: the gate has never run live; the counting script and protocol ship, the numbers come
  from one store's run with the flag on.
- **No lag numbers**: with extraction off `sync_turn` leaves no trace in the store, replay was ruled out, and a
  scheduler cannot be measured synthetically. What ships is the grace counter `_presence_grace` (debug-logged
  with the snapshot's turn); a late-sync event distinct from grace is **not implemented yet** (condition 3's
  lag readings need it) and readings themselves follow a live run only.
- Latency figures are fixed-input compute costs on one machine, not a live p95.

## Implementation checkpoints

- Shape of `messages` in `sync_turn`: is a tool row's `content` the raw result text (one `json.loads`) or a
  wrapper (two), and is `tool_name` carried — needed to recognise our own results;
- lock discipline: copy → parse outside → publish under lock with the generation check;
- config: `prefetch_presence_gate: false` plus the description line in the config block, per the existing
  `is_truthy_value` pattern;
- docs: the limits above and the cut/spill policies go into this file's final PR revision and the config
  description.
