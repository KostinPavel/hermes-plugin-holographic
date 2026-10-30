# Memory map (experimental)

> **Experimental.** The memory map is new (October 2026). Its JSON (`schema_version` 1), its rules for people and
> topics and its reading of contradictions may change between releases without a deprecation period, and it has been
> tried on invented data and test stores only. It is read-only and off unless you call it; read
> [Risks and limits](#risks-and-limits) before you put its output where anyone else can see it.

## What it is

`memory_map.py` turns **one** store into a JSON snapshot that a person can read through a renderer: today the
"Memory map" tab of [hermes-dashboard](https://github.com/itpartypattaya/hermes-dashboard), later perhaps a standalone
page. It answers the questions a person has about what the agent remembers about them: which facts, who said each one
and how much it is trusted, what changed or was forgotten this week, which facts contradict each other and which side
search leans on, what the nightly review (hermes-dreaming) is asking about.

It is meant for the **owner of the memory** (the person or family the profile serves) and for whoever looks after the
agent. It is not a tool for the agent: no tool calls it, and the provider behaves exactly as without it.

## How to run it

Nothing calls the export unless you do: there is no setting to turn on.

```
python <plugin dir>/memory_map.py --store PATH [--hermes-home DIR] [--tz ZONE] [--now ISO] [--config YAML]
                                 [--max-facts N] [--max-entities N] [--json]
python -m <package>.memory_map --store PATH ...       # the same; needs Hermes core importable (the package's __init__ loads it)
```

- `--store`: the store file — `memory_store.db` of the profile, or a scope's `.db`. One per run.
- `--hermes-home`: also read the nightly review's files under this directory (see [Dreaming](#dreaming-files)).
  Without it, none are read.
- `--tz`: the zone of the dates, an IANA name or `+HH:MM` (UTC by default; a Windows Python without the `tzdata`
  package knows only the offsets).
- `--config`: Hermes `config.yaml`, so that "which side search leans on" uses the search settings in force (trust
  floor, decay half-life, reranker). Without it the plugin's defaults are assumed and `SNAP.ranking.source` says so.
- `--max-facts` / `--max-entities`: the size limit, 1000 facts and 300 names by default; `0` lifts it (see
  [Size](#size)). A negative number is an error (exit 2).
- `--now`: the moment of the snapshot (default: now). Prints JSON to stdout; exit 2 when there is no store file
  (nothing is created), 1 when the store cannot be read (only the kind of the error is printed, never a stored value).

From Python: `export_map(store_path, hermes_home=None, tz=None, now=None, config=None, max_facts=1000,
max_entities=300) -> dict` — the same dict (`None` for no limit).
`python <plugin dir>/memory_map.py` runs without Hermes core on the path; the dashboard calls it that way, on a
schedule (see its README).

## What the map shows

- **Facts** in force and the ones that ended, each with its text, who said it (the author the store recorded) and in
  what kind of chat, its origin (said / confirmed / inferred / unknown), its trust as a number and a word (weak,
  normal, strong), when it was first recorded and when its current text was, its versions (what it said before and
  until when), the names linked to it, confirmations by a person, how often it was recalled and marked helpful.
- **Weak basis**: the provider's own `review()` reasons (low trust, never recalled, an inference nobody confirmed, a
  near-duplicate) and "no source" for a fact whose origin is unknown — except records written before origins existed.
- **Names** (entities) with their aliases, whether a person declared them or they were guessed, and a type: person,
  topic, or "not determined" (see [Entity types](#entity-types)).
- **Contradictions** the provider's `contradict` finds between visible facts: the words that tell the two apart and
  which of the two search puts first (see [Contradictions](#contradictions)).
- **Forgotten** facts as their number and date only; **merged** copies under the fact they were merged into.
- **The week**: what was added, changed, corrected, ended, confirmed, forgotten and merged in the 7 days before the
  snapshot, and the night of the nightly review.
- **Waiting for a person**: open merge questions, memory entries the night found in conflict with a fact, "is this
  still relevant?" questions about long-term memory, and how many writes to long-term memory wait for approval (counts
  only, never their text).

## What the map does not show

- **Another store.** One export, one store: the stores of other chats or scopes are not opened, not even counted.
- **Forgotten text.** A forgotten fact is its number and the date; the store no longer holds its text, keywords or
  names, and anything of the night's files about it (its numbers included) is dropped.
- **Corrected text.** A version that was corrected never held: it has no text (`text: null`).
- **What the safety screen keeps out of answers**: a credential, an instruction to the assistant, hidden markup. Such a
  fact is listed under `HIDDEN` with its number, date and the reason only; such an old version of a visible fact has
  no text. Names, aliases, categories, user names and the night's texts pass the same screen.
- **Writes waiting for approval**: counted, never shown (they are not memory yet).
- **How memory was used**: the usage journal (`experience`) is not exported in this version (`ATTEMPTS` is empty),
  nor are "last recalled" dates.
- **Anything new since the snapshot**: the map is a picture of one moment (see the risks).

## Privacy

- **Read-only in fact.** The file is opened `mode=ro` with `PRAGMA query_only`; no migration, no model, no network.
  The provider's own read code (history, origins, entities, `review`, `contradict`) runs over that connection, so any
  write it attempted would fail instead of happening. The store file is byte-identical afterwards (same hash, same
  modification time), on current and on older schemas, from the function and from the command line — tested.
- **One store, one scope.** See above; and the night's files are checked piece by piece (next section), so a chat's
  map never receives the profile's long-term memory.
- **Forgotten = number and date; corrected = no text; screened = no text.** As above, tested on every export path.
- **The output itself is personal data.** The JSON holds the texts of the facts and the names in them. Where it goes
  is up to the caller: see the risks.

## Dreaming files

With `--hermes-home`, the export also reads three things hermes-dreaming leaves there: `cache/dream-proposals.json`
(merge questions), `cache/dream.json` (the night's report) and `pending/memory/` (writes to long-term memory waiting
for approval). Whose each piece is, is checked separately, never inferred from another piece:

- the **report** (its `MEMORY.md` entries, its conflicts, the night's stats) and the **approval queue** belong to the
  profile store — `<home>/memory_store.db` — and are read only for that store. For any other store they are not even
  opened (`files: profile_only`), whatever the other files say;
- a **merge question** is this store's only when it names this very file (`store`, an absolute path, as hermes-dreaming
  writes it: the provider's own `database`); one without a `store` is the profile store's. A relative `store` proves
  nothing — read from the folder the export runs in, it would name whatever store lies there — so such a question is
  left out (`invalid`, `store_relative`). Two spellings of one file are one store (`..`, links, letter case on
  Windows). A question that names a chat's store does not make the report or the queue that chat's;
- a **report item that names another store** (a chat's own fact or entry; hermes-dreaming names a chat's database by
  its key) is that store's, in every section of the report (`conflicts`, `md_decays`): neither its text nor its fact
  number reaches this map;
- every **text of a night record** that reaches the JSON passes the provider's screen, not only the entry: a MEMORY
  conflict's numbers must be what hermes-dreaming writes there (a number, date or time as text, `month:MM`,
  `weekday:N`) and its negations short strings the screen lets through; a stat's name must be a code. Anything else
  leaves the record out (`invalid`, with the field). A number that is not finite (NaN, Infinity) is never written;
- a **date no calendar holds** (a queue record's `created_at: 1e100` or NaN, a moment past the year 9999 once moved to
  UTC) is that record's problem: the queue counts the record without its date and lists it in `invalid`; a merge
  question's date is null; a report whose `generated_at` is not an ISO moment (or has text after one) shows no night
  and is listed;
- a missing, unreadable or wrongly shaped **file** is a state (`absent | unreadable | unexpected`), never an error; a
  wrongly shaped **record** (a list where a record was expected, a store that is not a path, a fact number that is not
  a number) is left out and listed in `SNAP.dreaming.invalid` as `{file, id, why}` — the rest of the map, the night's
  good records included, is there.

`SNAP.dreaming.read` is false with `reason: not_requested` (no `--hermes-home`) or `other_store` (a store of another
scope, and no merge question names it).

## One snapshot

All reads from the store happen in **one read transaction**, opened before the first read and always ended (a
rollback: nothing was written) when the export ends, also on an error. The provider's read code joins that
transaction and never commits it; the export checks at the end that the transaction is still open and
raises otherwise. So a `forget`, a `change` or a merge that another process makes while the export runs is either all
in the map or not at all — never the fact's text next to its tombstone.

What the transaction does **not** cover: the night's files (`cache/*.json`, `pending/memory/`) are read separately,
right after the store. If the nightly review writes them during an export, the map can pair a store of one moment with
a report of the next night; a question about a fact forgotten in between is dropped all the same (the check is done
against the store's snapshot).

In rollback-journal mode (`journal_mode=DELETE`: what Hermes keeps on a SQLite it deems WAL-reset-vulnerable, and on
network file systems) a reader holds the database for writers: during an export, a write by the agent waits for it to
finish, up to the provider's 10-second busy timeout, and fails after that. On a synthetic store of 3,000 facts an
export took 0.7 s on a laptop (i5-1135G7); a much larger store, or a slow disk, makes that window longer. In WAL mode
writers are not held.

## Risks and limits

- **Experimental**: formats and rules may change; tried on invented data and test stores only.
- **Personal data in the output.** The JSON — and any page made from it, such as the dashboard's HTML — shows what
  the agent knows about people: names, family, plans, habits. Keep it behind a password (the dashboard's basic auth),
  never on a public page, never in a public repository or a chat. Making it public is a leak of everything in memory.
- **"Person" or "topic" is a narrow rule, not understanding.** A name is a person only when it is a recorded chat
  author's name or a word from a short closed list (Russian and English: family, friend, colleague, nanny, teacher,
  doctor, Mr/Mrs/Ms/Dr) stands right beside it in a stored fact. Everything else is "not determined": most names will
  be. Other languages and other roles are not recognised. A wrong "person" is possible when such a word happens to
  stand next to a name it does not describe.
- **"Which side search leans on" is true only in a narrow case.** It compares the part of the score no question
  changes — the trust floor, trust and age decay — so it holds only for a question that fits both facts' words
  equally. A question that fits one fact's words better can put the other first. With the reranker on the answer is
  "not said" (it orders by text alone); with search by meaning the rule still holds for equal fit. Without `--config`
  the plugin's defaults are assumed, which may not be the settings in force.
- **Age is counted as search counts it**: from when the fact's current text was recorded (0.23.0), at the snapshot's
  moment; new keywords, tags or a vote do not make a fact young. A store without history counts from its last change.
- **Contradictions are the provider's heuristic** (`contradict`: shared names, other numbers, negation), not a judgment;
  some real contradictions are missed and some pairs are not contradictions.
- **The night's files are outside the snapshot** (see above).
- **The picture ages.** It is the store at one moment; the dashboard rebuilds it on its cron schedule (every 30 minutes in
  its example crontab), so a page can be up to that old: a fact forgotten a minute ago may still show its text until the next
  rebuild. Copies already rendered, saved or sent elsewhere are not reached by a later `forget`.
- **The profile store is recognised by its path**, `<home>/memory_store.db`. A profile whose `db_path` points
  elsewhere is treated as "another store": its night's report and queue are not shown.
- **Writers wait during an export in DELETE journal mode** (see "One snapshot").
- **No usage journal**: nothing says when a fact was last used in an answer; "recalls" counts how often it was handed
  to the model, not whether it helped.
- **"Stale" names** (provisional for 60+ days) are counted by the clock of the run, not by `--now`.

## Size

The map is read as a page, and a page made from it (the dashboard's) holds every fact, its versions and a card per
name; every visit downloads it. On a test store of 10,000 facts and 1,000 names the dashboard page grew to about 26 MB
and took 8 s to draw per language. So the export keeps at most **1000 facts and 300 names** unless told otherwise:

- Facts a person is asked about or that changed lately come first: both sides of each contradiction, then what the
  nightly review asks about, then the facts of the week's feed (newest first). Then current facts, then closed ones,
  each newest first by when the current text was recorded.
- Names with the most facts stay; a kept fact and a contradiction (`shared`) list only kept names. A name's `facts`
  is the whole store's count, so a name card can count more facts than the map shows.
- `SNAP.counts` stay the whole store's; `SNAP.limits` says what was cut:
  `{"facts": {"max", "total", "shown"}, "entities": {"max", "total", "shown"}}` (`max` null: no limit).
- A cut fact has no card and is not in `FACTS`. Records that name it keep their own fields, which the privacy rules
  already allow: a merge keeps the merged copy's text, a nightly-review question its texts, a feed line the number.
  A renderer must expect a fact id it cannot find. If the questions alone name more facts than the limit, some of
  them are cut too.
- Versions of one fact, forgotten numbers and merges are not limited.

By the 10,000-fact measurement the default limit makes a dashboard page of roughly 3 MB per language (an estimate:
not measured at exactly 1000 facts).

## The JSON (`schema_version` 1)

Top-level keys, in this order: `schema_version, SNAP, ENT, FACTS, CONFLICTS, PENDING, TOMBS, MERGES, HIDDEN, FEED,
ATTEMPTS` — the names of the design mockup's demo data. Lists are sorted (facts and entities by id, conflicts by score,
the week newest first), so the same store and `now` give the same bytes.

- `SNAP`: `made` (ISO with offset), `tz`, `store` (file name only), `schema` (meta versions), `counts` (current, ended,
  merged, forgotten, corrected, hidden, entities, provisional, legacy), `week_from`, `excluded`, `ranking` (the search
  settings contradictions are ordered by: `source` defaults | config | absent | unreadable, `min_trust`, `half_life`,
  `rerank`, `meaning`), `experience` (`included: false` in this version), `dreaming` (`read`, `reason`, per-file
  state `ok | absent | unreadable | unexpected | profile_only`, `invalid` [{file, id, why}]), `limits` (see
  [Size](#size)).
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
- `SNAP.dreaming.invalid[].why`: `id` (a proposal number the map cannot read; `id` is then null), `record` (not an
  object), `store` (not a path), `store_relative` (a merge question's store that is not an absolute path), `keep` /
  `copy` (not an object), `fact_id` (not a number), `list` (a report section that is not a list), `conflicts` /
  `entry` / `index` (an item's own field of another shape), `fact_numbers` / `memory_numbers` / `fact_negations` /
  `memory_negations` (not the numbers or phrases hermes-dreaming writes, or text the screen refuses), `generated_at`
  (the report's moment is not one), `key` (a stat's name is not a code; the name is not repeated), `created_at` (a
  queue record's date no calendar holds). `file` is `proposals`, `report` or `queue`; `id` is the proposal number,
  the report section and index (`conflicts[2]`, `md_decays[0]`, `conflicts[1].conflicts[0]`), `generated_at`,
  `stats`, or the queue file's name (null when it is not a plain one).

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
- otherwise trust × decay decides between two facts a question fits equally (`why: weight`). The decay is search's own
  (`FactRetriever.decays`, the same code `_rerank` calls): `0.5^(age / temporal_decay_half_life)`, the age counted from
  when the fact's current version was recorded (`fact_versions.recorded_at`; `updated_at` in a store without history),
  at the snapshot's moment; no decay when the half-life is 0, the default. A question that fits one fact's words
  better can still put the other first; the renderer says so;
- `on: null` with `why`: `tie` (equal weights), `neither_found` (both below the floor), `rerank` (the reranker is on:
  it orders by text alone, so trust says nothing), `settings_unknown` (the config could not be read).

`sides` holds per fact `{trust, found, decay}`. The settings are the plugin's defaults unless `config` is given (a
config.yaml path, read with Hermes' own loader `hermes_yaml`, or PyYAML outside Hermes, or the section as a dict): `SNAP.ranking` says which.
Search by meaning (`meaning: true`) blends relevance before trust multiplies it, so the rule holds there too.

`differ` is what tells the two apart, from their own words only (nothing is generated): `a` / `b` the words of each the
other lacks (by their keys, so word forms count as one word; function words and the linked names left out; at most
8), `negation` true when one is the negation of the other (contradict's own reading). `since` is the day the later of
the two current versions was recorded: the store keeps no date for a contradiction itself.

## Not in this version

The usage journal (`experience`), roles of a person, "last recalled" dates, a diff against the previous snapshot, a
standalone page. The renderer is the hermes-dashboard tab.
