# A token budget for recalled facts (0.21.0)

Before each turn the provider injects up to five facts (plus the top facts of people the message names) as one
block. On a small context window, or a store full of long facts, that block can cost more than it is worth.
`prefetch_max_tokens` limits it:

```yaml
plugins:
  hermes-memory-store:
    prefetch_max_tokens: 100   # approximate tokens; 0 or unset = off (the default)
```

## What it guarantees

- **The whole block counts**: the `## Holographic Memory …` header, every fact line with its `#id [trust]`,
  `(by meaning)`, `(inferred)` and `recall_source` suffix, and the `Memory attempt:` line when the experience journal
  is on. The fact-count limit still applies; the budget is an extra limit on top of it.
- **The budget never cuts a line.** A fact whose line does not fit is left out whole: a cut can drop the "not", the
  condition or the date and turn it into a different claim. The next, shorter fact may still fit, so the facts that
  go in are not always a prefix of the ranking, but they keep its order. This is a guarantee about the budget, not
  about every line: the older 300-character cap on a prefetch line (it predates the budget and applies with the
  budget on or off) still shortens a longer fact to 299 characters and "…" — see the stand below. `fact_store search`
  returns such a fact in full.
- **A left-out fact is not "recalled"**: it gets no recall count, no journal receipt, and it is not held back by
  `prefetch_repeat_turns`, so the next turn can bring it.
- **Nothing fits** → no block at all, as on a turn with nothing to recall.
- **Reported without text**: `fact_store stats` shows
  `prefetch_budget: {max_tokens, last: {kept, dropped, tokens}, dropped_total}` while the budget is on; a debug log
  line gives the counts and the left-out fact ids. No fact or message text is logged.
- **Off is off**: unset or `0` injects byte for byte what 0.20.x did, and `stats` has no `prefetch_budget` key.
- A value that is not a whole number, or is negative, is a config problem named in the system prompt; memory stays
  off for the session until it is fixed, as with every other numeric key.

## The token count is approximate

No tokenizer ships with the plugin and none is added as a dependency, so tokens are estimated per character:
ASCII ≈ 4 characters per token (the same rough rule Hermes core uses), a wide character (CJK, kana, Hangul, emoji)
≈ 1 token, any other non-ASCII character (Cyrillic, Greek, Thai, accented Latin) ≈ 2 characters per token. Each
class is rounded up per line, so the sum over the lines is never below the estimate of the whole block. For common
tokenizers this errs on the high side for English and Cyrillic; it is not the exact count of any model. Leave a
margin if a hard limit matters.

## Why the default stays off

A non-zero default would silently drop facts for everyone whose facts are long, and the right size depends on the
model's context, which the plugin does not see. Measured on the stand below, the unlimited block averages ~86
approximate tokens, which is already small next to a typical system prompt; a budget is for small contexts and
stores with long facts, chosen by the person who runs it.

## Measurement stand

`tools/provenance_probe.py --set budget --budgets 0,150,100,60` over `tools/budget_set.py`: 38 invented facts
(six of them changed once, six long ones full of conditions), 40 questions in six slices — direct, paraphrase,
namesake, changed fact, no answer, long fact — split into dev (23) and a held-out test (17). No network: rerank and
embeddings are off. Rows: `results/provenance_budget_all.jsonl`, table: `results/provenance_budget_all_summary.json`.
Latency is the best of three runs per question on a Windows laptop (i5-1135G7) under load; read it as "no measurable
overhead", not as an absolute number.

| budget | split | hit (all labels) | lost vs unlimited | noise | extra facts / q | stale | cut lines | tokens avg / max | left out | ms p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| off | dev | 18/19 | — | 2/4 | 1.70 | 0 | 3 | 86 / 142 | 0 | 14.4 |
| off | test | 12/14 | — | 0/3 | 1.53 | 0 | 3 | 85 / 155 | 0 | 15.1 |
| 150 | dev | 18/19 | 0 | 2/4 | 1.70 | 0 | 3 | 86 / 142 | 0 | 14.0 |
| 150 | test | 12/14 | 0 | 0/3 | 1.47 | 0 | 2 | 79 / 142 | 1 | 16.1 |
| 100 | dev | 18/19 | 0 | 2/4 | 1.39 | 0 | 0 | 65 / 97 | 7 | 13.5 |
| 100 | test | 11/14 | 1 | 0/3 | 1.29 | 0 | 0 | 63 / 97 | 5 | 13.8 |
| 60 | dev | 15/19 | 3 | 1/4 | 0.83 | 0 | 0 | 41 / 50 | 23 | 13.8 |
| 60 | test | 9/14 | 3 | 0/3 | 0.88 | 0 | 0 | 44 / 50 | 14 | 14.5 |

What it shows:

- **100 tokens** cuts the average block by a quarter (86 → 64) and extra facts by a sixth; the one labelled fact it
  loses (test `l06`) is the 349-character Kestrel brief, whose capped line alone costs ~79 tokens and comes after two
  shorter Kestrel facts.
- **60 tokens** is below one long fact plus the header (a 210-240-character fact costs ~55-63 tokens, the header
  15): every long-fact question loses its answer. Below about 80 tokens long facts are not recalled before the turn
  at all; they stay reachable through `fact_store search`.
- **Old versions never come back** (`stale` 0 everywhere): changed facts are recalled at their current text with or
  without a budget.
- **The pre-existing 300-character line cap does cut facts**: without a budget, `l06` counts as a hit, but its line
  ends in "…" before the asked part (the logo and its date). The budget never adds such a cut — a line that does not
  fit is left out — but it does not lift the old cap either. That cap is unchanged in 0.21.0.
- Misses that do not depend on the budget: three paraphrase questions with no shared words (lexical recall alone;
  search by meaning is off on the stand). Noise: a message naming a person recalls that person's facts by design
  (`x01`), and "bank card PIN" shares "bank" with a fact (`x03`).

The set is small and English only; it shows the trade-off, not a recommendation for every store.
