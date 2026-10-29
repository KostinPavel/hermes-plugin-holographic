# Memory map export (data layer)

`memory_map.py` turns ONE store into a JSON snapshot a person can read through a renderer (a dashboard tab, later a
standalone page). It is read-only and has no switch: nothing calls it unless you do, and the provider behaves exactly as
without it.

```
python -m <package>.memory_map --store PATH [--hermes-home DIR] [--tz ZONE] [--now ISO] [--config YAML] [--json]
python <plugin dir>/memory_map.py --store PATH ...      # runs without Hermes core on the path
```

From Python: `export_map(store_path, hermes_home=None, tz=None, now=None, config=None) -> dict`. A missing store file raises
`FileNotFoundError` (the CLI exits 2); nothing is created. `tz` is an IANA name or `+HH:MM` (UTC by default; a Windows
Python without the `tzdata` package knows only the offsets).

## What it guarantees

- **Read-only in fact.** The file is opened `mode=ro` with `PRAGMA query_only`; no migration, no model, no network.
  The provider's own read code (history, origins, entities, `review`, `contradict`) runs over that connection, so any
  write it attempted would fail instead of happening. The store file is byte-identical afterwards.
- **One store.** Another scope's file is never opened. The dreaming files are read only with `hermes_home`, and only
  when they are about this store: it is `<home>/memory_store.db`, or a proposal names it. Otherwise
  `SNAP.dreaming.reason` is `other_store` and nothing of them is shown.
- **Forgotten = number and date.** `TOMBS` holds `{id, d}`; the store keeps no text, keywords or names of it, and a
  dreaming item about a forgotten fact is dropped whole (its numbers are its text too). A memory entry from the dream
  report that the provider's tombstone check recognises as a forgotten text is dropped as well.
- **No text where the provider shows none.** A `corrected` version has `text: null`. A text the provider's screen keeps
  out of answers or away from the network (`_recall_problem` or `_secret_in`: a credential, an instruction to the
  assistant, hidden markup) is not shown: such a fact is only in `HIDDEN` (`{id, d, why, status}`), and such an old
  version of a visible fact has `text: null, hidden: true`. Names, aliases, categories, user names and dreaming texts
  pass the same screen. A name only hidden facts give is not listed.

## The JSON (`schema_version` 1)

Top-level keys, in this order: `schema_version, SNAP, ENT, FACTS, CONFLICTS, PENDING, TOMBS, MERGES, HIDDEN, FEED,
ATTEMPTS` — the names of the design mockup's demo data. Lists are sorted (facts and entities by id, conflicts by score,
the week newest first), so the same store and `now` give the same bytes.

- `SNAP`: `made` (ISO with offset), `tz`, `store` (file name only), `schema` (meta versions), `counts` (current, ended,
  merged, forgotten, corrected, hidden, entities, provisional, legacy), `week_from`, `excluded`, `ranking` (the search
  settings contradictions are ordered by: `source` defaults | config | absent | unreadable, `min_trust`, `half_life`,
  `rerank`, `meaning`), `experience` (`included: false` in this version), `dreaming` (`read`, `reason`, per-file
  state `ok | absent | unreadable | unexpected`).
- `ENT`: `id, name, type (person | topic | null), type_by, type_fact, declared, aliases, text_aliases, standing
  (established | provisional), stale (provisional 60+ days, as review), since, facts` — see "Entity types".
- `FACTS` (current and ended; merged copies are under their keep): `id, text, ents, cat, origin (said | confirmed |
  inferred | unknown), trust (2 decimals), trust_level (weak < 0.45 ≤ normal < 0.65 ≤ strong), status, from, to,
  to_exact, rec, first, via (tool | auto_extract | mirror | legacy), chat (the chat type; chat names are not stored),
  by, conf [{d, by}], recalls, helpful, last (null: not stored), versions [{seq, status, text, from, to, to_exact, rec,
  closed, origin, via}], mergedFrom [{id, text, d, prop}], weak [codes], dup_of?`.
- `weak` codes: `review()`'s reasons (`low_trust` < 0.3, `never_recalled`, `unconfirmed_inference`, `near_duplicate`;
  review gives one reason per fact) and `no_source` — origin unknown, except records from before origins existed
  (`via: legacy`), which are not weak for that (decided 29.10.2026; `SNAP.counts.legacy` counts them).
- `CONFLICTS`: `contradict` pairs of visible facts: `id (C1…, internal: no tool takes it), a < b, score, shared
  (entity ids), since, differ, relies` — see "Contradictions".
- `PENDING`: `conflict` (one per `CONFLICTS` pair), and from dreaming: `merge` (open D-NN: `keep, copy, covered,
  extra, created, asked, current` — false when a side changed since), `memory_conflict` (a fact against a long-term
  memory entry: `fact, entry, *_numbers, *_negations`), `relevant` (`R-<index>`: "still relevant?"), `queue` (writes
  waiting for approval: counts and the oldest date, never their text).
- `TOMBS` `{id, d}`, `MERGES` `{from, into, d, prop, text}`, `HIDDEN` `{id, d, why: credential | instruction | markup,
  status}`.
- `FEED`: the 7 days before the snapshot: `new`, `chg` (`how: changed | corrected`), `end`, `conf`, `del` (id only),
  `mrg`, `night` (dream report: alert kinds and numeric stats only).
- `ATTEMPTS`: empty in this version (the usage journal is not exported yet).

## Entity types

The store has no types: an entity is a name. `type` says what the store itself shows about it, and is null otherwise —
it is never guessed from a capital letter, since a month, a brand and a pet are capitalized too.

- **person** when (`type_by`) `speaker`: the name or an alias is, by `entity_key`, the author's name the store recorded
  with a version of any fact (`src_user_name`); or `relative_or_role`: in a visible fact linked to it, a word for a
  relative or a role stands right before the name («дочери Варе», "my daughter Varya", "Mr Smith"), or after it with
  only "is / my / our…", «моя / наша / это…» between ("Varya is my daughter", «Варя — наша дочь»; not "and" / «и»).
  The name is found in any case form, as recall finds names (`text.name_keys`: «Вари», «Варе» are «Варя»); a Latin
  name as written or with 's. The word list is short and closed (Russian and English: family, friend, colleague,
  nanny, teacher, doctor, Mr/Mrs/Ms/Dr); a possessive is not such a word ("my daughter's school" names no one).
  `type_fact` is the fact that showed it.
- **topic** when the name has a form no person's name has (`type_by`): `digit` ("Лицей №3", "iPhone 15"),
  `identifier` (a dot, colon, slash or underscore inside a word), `mixed_case` (a word that starts in lower case and
  has a capital later: "iPhone"), `acronym` (one word of two or more letters, all capitals: "USDT").
- **null** otherwise, and when both a person's sign and a topic's are there (`type_by: both`). A pet's name, a
  child's name nobody put a relative's word beside, a brand written as a plain word: all null. A renderer shows them as
  "not determined", not under people.

## Contradictions

Each `CONFLICTS` pair says which of its two facts search puts first now (`relies`), by the part of search's score no
question changes (`FactRetriever.search` → `_rerank`: relevance × trust × decay):

- a fact below the trust floor (`min_trust_threshold`, 0.3 by default) is never found: the other one is `on`
  (`why: floor`);
- otherwise trust × decay (`0.5^(age / temporal_decay_half_life)` from `updated_at`, at the snapshot's moment; no
  decay when the half-life is 0, the default) decides between two facts a question fits equally (`why: weight`). A
  question that fits one fact's words better can still put the other first; the renderer says so;
- `on: null` with `why`: `tie` (equal weights), `neither_found` (both below the floor), `rerank` (the reranker is on:
  it orders by text alone, so trust says nothing), `settings_unknown` (the config could not be read).

`sides` holds per fact `{trust, found, decay}`. The settings are the plugin's defaults unless `config` is given (a
config.yaml path, read with PyYAML when it is importable, or the section as a dict): `SNAP.ranking` says which.
Search by meaning (`meaning: true`) blends relevance before trust multiplies it, so the rule holds there too.

`differ` is what tells the two apart, from their own words only (nothing is generated): `a` / `b` the words of each the
other lacks (by their keys, so word forms count as one word; function words and the linked names left out; at most
8), `negation` true when one is the negation of the other (contradict's own reading). `since` is the day the later of
the two current versions was recorded: the store keeps no date for a contradiction itself.

## Not in this version

The usage journal (`experience`), roles of a person, "last recalled" dates, a diff against the previous snapshot, the
renderers themselves.
