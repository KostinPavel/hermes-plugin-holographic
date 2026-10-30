# The presence gate: design note (proposed for 0.26.0)

Working note for the presence-gate PR, mapped line by line to the acceptance conditions in issue #59
(conditions 0–8 of comment 6523787794). Status: design agreed in the thread, code not started.
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
either visible in the current-generation snapshot or was injected after that snapshot was taken (grace window).
Everything else follows today's rules: not-injected facts, other stores' `#N`, foreign blocks, prose — nothing
changes for them.

## State

- `_injected[key][revision]` stores, per injection event, **the exact line the plugin emitted** — cut, collapsed
  and suffixed as built — plus the turn. A list per revision, because `(by meaning)` depends on the retrieval
  path of that event: the same revision can be emitted with and without the suffix, and each event must match
  its own row.
- A **compaction generation** counter advanced by `on_pre_compress`. With the gate on the hook does not clear
  `_injected`; it marks the snapshot stale (condition 0).
- `_snapshots[key] = (generation, visible)` filled by `sync_turn`: copy the live `messages` at the very start of
  the method, parse **outside** the provider lock on that copy, publish the result under the lock with a
  generation check (the maintainer's implementation note). The snapshot counts for the current generation only
  if it shows it belongs there — it contains that compaction's summary row (`_compressed_summary = 1`,
  core `state.db`). Otherwise it is stale and the gate falls back to today's behaviour; the same fallback covers
  `messages=None`, an empty list and a sync that never ran.

## Two zones, two representations — no cross-form search

| Zone | What the context physically holds | Artifact stored per event | Match |
|---|---|---|---|
| Recall block (`<memory-context>`) | **only the cut form**: collapsed, ≤300-char cap, suffixes | the emitted line, verbatim | whole line ↔ whole line inside a `<memory-context>` block |
| `fact_store` JSON results | **only the full form**: `content`, uncut | `(fact_id, content)` of the revision | `json.loads` the result, compare the pair |

Neither form ever appears in the other zone, so nothing is searched in both versions; the gate never
reconstructs a representation it did not produce itself.

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
`hooks.output_spill` (default cap 10 000) replaces an oversized hook payload with `preview_head`/`preview_tail`
plus a path, so a fresh snapshot may show the preview instead of the block; treating that as "not visible" would
hide and re-show the facts on alternating turns. Order verified: `_remember` runs inside `prefetch`, the core
spills later in `turn_context` (`agent/turn_context.py`, `spill_if_oversized`). Test: no flip-flop.

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

## Tests (condition 7, plus design-specific ones)

Built the way the core builds context (`build_memory_context_block` wrapped into `api_content`):

1. compacted head, kept tail — the tail's fact is not re-injected, the head's one is; **this test fails on
   current `main`** (stated in the PR);
2. another provider's block and a user message containing `#5 [` — no effect;
3. two `user`-scope stores with the same fact id — independent;
4. a revised fact — old text visible does not suppress the new revision (`test_a_reworded_fact_is_recalled_again`
   stays meaningful);
5. prefetch before the snapshot arrives — grace window, not silence;
6. spilled block — the policy above, no flip-flop;
7. a line emitted with `(by meaning)` matches only its own row, never the suffix-less one;
8. a legacy `\uXXXX` tool result still counts as visible (pre-#62 messages);
9. `export` output does not count as visibility;
10. `messages=None` / empty / stale snapshot — byte-identical behaviour, pinned.

## Metrics

Definitions: a duplicate = a `(session, fact_id)` key whose fact shows in ≥2 rows live now (`active = 1`),
lines read structurally (`<memory-context>`, our `- #N [trust]` shape only). Snapshot of this store, 2026-10-30:
**115 keys in 11 sessions** (101 in 10 the same morning — the store grew); of them **5 row-byte-identical twins
(kept in, per the maintainer)**, **106 identical line text**, **9 differing text under one `#N`** (a revision or
two stores sharing an id — the replayed row does not name its store, so pairs are grouped by text and the 9 stay
a labeled residual); cause between first and last live row: **8 compaction-preceded, 107 other** — window expiry
in uncompacted context dominates ("preceded by", not "caused by"). The same script runs after the gate for the
after number. The PR also reports `sync_turn` lag (how often a sync starts after the next turn's prefetch has
already run, by how many turns) and prefetch/sync latency, as asked.

## Implementation checkpoints

- Shape of `messages` in `sync_turn`: is a tool row's `content` the raw result text (one `json.loads`) or a
  wrapper (two), and is `tool_name` carried — needed to recognise our own results;
- lock discipline: copy → parse outside → publish under lock with the generation check;
- config: `prefetch_presence_gate: false` plus the description line in the config block, per the existing
  `is_truthy_value` pattern;
- docs: the limits above and the cut/spill policies go into this file's final PR revision and the config
  description.
