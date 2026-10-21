# Maintenance notes — hermes-plugin-holographic

**Status: maintained by [itpartypattaya](https://github.com/itpartypattaya)** (since 2026-10-10). This repository
started as Nous Research's standalone copy of the `holographic` memory provider bundled in
`NousResearch/hermes-agent` under `plugins/memory/holographic/`, published so someone could take it over
([original handoff note](https://github.com/NousResearch/hermes-plugin-holographic/blob/main/HANDOFF.md),
[takeover request](https://github.com/NousResearch/hermes-plugin-holographic/issues/1)).

## Catalog and migration

The plugin is submitted to the Hermes plugin catalog as `plugin-catalog/holographic.yaml` under the name
`holographic`. That exact name matters: once core drops its bundled copy, `hermes update` (and agent start) install
the catalog plugin of the same name for users with `memory.provider: holographic`
(`hermes_cli/memory_provider_migration.py`, `_LEFT_CORE`), keeping their config and data.

Releases are tags `vX.Y.Z` matching `version` in `plugin.yaml` and `pyproject.toml`; the catalog entry pins the
tag's commit by `sha:` and `version:`. A new release reaches users only through a SHA-bump PR to hermes-agent.

## Differences from the in-tree copy

- Self-imports are relative so the package loads from `~/.hermes/plugins/holographic/` under the loader's synthetic
  namespace.
- `pyproject.toml` is the dependency authority (no `tools.lazy_deps` calls: they pinned an old version in the
  lazy-deps registry and downgraded newer installs, hermes-agent#86992).
- Behaviour changes since 0.2.0 are listed in [CHANGELOG.md](CHANGELOG.md).

While a Hermes install still carries the bundled `plugins/memory/holographic`, that copy wins on name and this plugin
is dormant; it takes over once core drops the bundled directory.

## Tests

`tests/` holds the in-tree tests (`tests/plugins/memory/test_holographic*.py` in hermes-agent) with the import path
switched to the package `tests/conftest.py` loads from this repo, plus `test_holographic_v020.py`. Run locally with
`HERMES_AGENT_REPO=~/.hermes/hermes-agent PYTHONPATH=~/.hermes/hermes-agent python -m pytest -q`, against a core as
new as CI's (`main`): a checkout older than the plugin lacks modules it imports (`hermes_state_wal`, `hermes_yaml`),
and the core itself needs `pyyaml` and `ruamel.yaml`. Lint: `ruff check .` (rules in `pyproject.toml`). The checkout
directory name must be a valid Python identifier (CI uses `plugin`): pytest imports the repository root as a package.
CI runs the suite on Linux, macOS and Windows against hermes-agent `main`, and `hermes plugins validate
--install-deps`, the catalog's admission check.

## Languages

`text.py` holds everything per language: function words (`STOP`), stemmer choice (`_STEMMERS`, `_DEFAULT_LANG`,
`_SCRIPT_LANGS`, `LATIN`), greetings, negators, claim words, folding (`_FOLD`); for the n-gram scripts the function
characters (`CJK_FUNCTION`, Japanese particles included), Thai function words (`_TH_FUNCTION`) and filler runs
(`FILLER_RUNS`). Synonyms come from the agent's `keywords` (stored after a `keywords:` line in `tags`, see
`_split_tags` / `_join_tags` in `__init__.py`); the keyword benchmark generated them with a separate agent that never
saw the questions, and the example in the `keywords` description must never be a benchmark fact (one batch was
discarded for that). To add a language: its function words (they
also decide which Latin language a text is), its stemmer, greetings and negators; then a main set and an unseen set in
`tests/test_holographic_languages.py` style, measured before the change. Arabic, Indic scripts and Indonesian go
through the word scan in `retrieval.py` (`_texts`, `_scan_candidates`), not FTS5; keep that in mind for any script
whose words FTS5 splits or whose inflection is a prefix. The README "Languages" section states how every number was
obtained; keep it that way.
A rebuilt German table is compared with the shipped one over every German noun, adjective, verb lemma and verb form
of the run (de_parts with both), not nouns alone: 0.17.0 was built without `forms_de_*.csv`, lost 1,530 verbs and cut
1,515 verb forms (`gehörte` as `geh|orte`), and a nouns-only comparison showed nothing (review 0.17.0, Fable; the
build now stops without the forms).

## Entities

Names come from a fact's text (`store._extract_entities`: capitalized runs, quotes, `X aka Y`) through
`_name_shaped`, or from someone (`entities=[...]`, `alias`, `rename`), which only `_valid_entity` checks. Any change
of the text rules that drops what they used to take must raise `_ENTITY_RULES` in `store.py`: a writer's first open
then rebuilds the index of every store up to `_AUTO_REBUILD_MAX_FACTS` facts (0.40 ms per fact measured, 0.16.0),
leaving declared entities, given and legacy aliases (`entity_aliases.source` other than `aka`) and their links
alone. The rebuild replays the facts in stored order, as adding them would: a text must never merge a name already in
use (that was a hijack: "Mallory Jones aka Alice Smith"). A rule for a language goes into `text.py` beside its other
data (`ZH_PHRASE_WORDS`, `ZH_FINAL_PARTICLES`), by words the language's own split gives, never by characters — a
character rule drops names that hold the character inside one word (别克, 网吧) — and is measured on real titles and
brands before it drops anything (平凡的世界, 我和我的祖国 hold function words). A phrase that only reads as a name is
not the gate's job: it stays `provisional` until a second fact or a person confirms it. Deleted names live in
`entity_tombstones` as keyed hashes; nothing may store a deleted name in clear, and an alias a text gave (`aka`) goes,
with the name it brought, when the last fact saying it goes. Reads use `name_key` / `entity_aliases` whenever the file
has them (`_keyed`), writes only after this process migrated (`_entry["migrated"]`, `_require_writer`).
Since 0.16.1 every rule reads text the same way in three places — `_extract_entities`, `_aka_pairs` and
`mentioned_entity_ids`: NFC (NFKC for mentions), code blanked (`_without_code`), words split at script seams
(`_script_pieces`; marks and modifier letters belong to their word), no name across a line break, a tab or the
separators `_mention_stretches` cuts at (but a dot inside a word or after an initial). A fix in one of them belongs
in the others, and every fix is measured on names that must stay (Node.js, nested lists, 平凡的世界) before it ships:
two fixes of the first 0.16.1 draft broke names that 0.16.0 found.
Since 0.16.2 a rule for the gate is measured on real names before it ships, not on a list from memory: Chinese labels
of films, series, games, books, companies, brands, apps and sites from Wikidata (26,308 in 0.16.2) against interface
strings from the zh-Hans translations of Grafana and Obsidian (6,584). A rule that loses real names to catch labels
goes to the standing instead (`_entity_rows`: a name whose letters have no case, only ever in plain quotes, stays provisional) —
that is how the verb rule of 0.16.2 was dropped (2.8 % of the names lost in its best form).

## Review

Every release before its tag, and every finding a second model (Codex, Fable, Antigravity) brings, goes through five
questions (from the review against ten principles of good code for AI agents, 2026-10-21; the plan and both reports
are in `plans/2026-10-21-*`):

1. Which requirement of this release justifies each new layer, parameter and dependency? A finding with no user and no
   failing case behind it is written down, not shipped.
2. Did removing a repetition turn into one function with many flags?
3. Are the existing contracts kept: names, arguments, answers, side effects?
4. Is a failure hidden behind an empty result, a caught exception or a success status? A catch at the boundary
   (a turn must not fail over memory) still says what happened, once, where someone looks.
5. Which checks really ran, with what result, and what was left unchecked? (`pytest` against hermes-agent `main`,
   `ruff check .`, the dream's screen comparison when the screen changed.)

## Open work

- Review of 2026-10-21 (`plans/2026-10-21-plan-pravki-10-principov.md`): waves 1, 3 and 4 are in `Unreleased`; a
  Codex review of that diff found two things, both fixed with a test (a non-numeric setting no longer quotes its
  value, `recall_error` is reset when another store becomes active); a second one three more (a failure is named
  by its kind, never its text: SQLite quotes a column's value; `entities_not_linked` gives positions, not the
  refused names; `recall_error` is reset on a session switch too). Left
  for a decision: the screen as its own `screen.py` without imports from the core (the dream would copy one file
  instead of walking the AST; do it together with the dream's 3.0), and a CI job on Python 3.11, the minimum in
  `pyproject.toml` (the core installs no runtime dependencies below 3.14, so the job needs its own set, or the
  minimum moves up). Not done on purpose: `memory_banks` and `remove` / `supersedes` (compatibility), the
  `_RU_DERIVATION` / `_EN_DERIVATION` rule branches (tests keep them as an option), `zip(strict=True)` (a
  behaviour change, not a tidy-up).

- Credential screen (0.17.2, issue #6): a name counts only when the credential word is glued to it by a hyphen and the
  whole word names a thing; a word that stands alone keeps its value whatever it looks like. Known false positives
  left on purpose (each fix loosens a security screen; review first): "the token counter.py module" (a file after a
  standalone word), "share the token-saver report" (any send request with a credential word), "перешли ключ от
  дачи" (a physical key; the dream skips it on its side), "Share grandma's secret pie recipe" (`secret` as an
  adjective), "fix/token-refresh-2". Lost on purpose in 0.17.4 (issue #6, second round): "pins card: 4821" (lower-case
  `pins` with a word after it is the verb), "(password): correct-horse" (two words after a closing bracket),
  "password-hunter2/" (a folder, as "password-hunter2.txt" since 0.17.2). The dream (hermes-dreaming, skills/dreaming/scripts/holo_screen.py) carries a
  verbatim copy of the screen: after any change here run its tools/vendor_screen.py, or its test and smoke go red.
  That copy runs on Python 3.9 and up: no possessive quantifiers or other 3.11 regex syntax in the screen (0.17.2
  used `\w*+` and the dream's copy failed to load there; 0.17.3, a test checks it).

- "X aka Y" (0.17.0): a declared name never resolves through an alias a text gave (split), alias / rename /
  delete_entity by one are refused, `split` separates any alias. Left open on purpose: a text that says Y still links
  to X (recall through the hint until someone splits); a known multi-word name a text mentions links a namesake's fact
  (`mentioned_entity_ids`); `entity_key` folds spellings together (`Bob-Brown` = `bob brown`); should automatic
  extraction (someone else's words) give aka aliases at all — watch a week of a real store first. Also open (0.17.1,
  review Fable): an entity row without `name_key` that an older version wrote beside an aka alias with the same key
  makes `entity_ids` give two ids, and a declared name then resolves to the smaller one; only with an older version
  writing the same file. An alias of unknown origin (`legacy`, before 0.16.0) that a text gave is not refused.

- Fact vectors after a turn on a slow endpoint (0.16.3, issue #5): batches follow the measured speed, checked with
  the reporter's numbers and a fake endpoint only, never against a real local model here. The catch-up runs on the
  core's one after-turn worker; if a large store on a slow server makes people raise `embedding_catch_up_seconds`
  far, a thread of the plugin's own (one at a time, stopped by `shutdown`) would free that worker — not done: it
  needs a lifecycle and tests that wait for it.

- Unsegmented scripts use a substring scan of character n-grams; an FTS5 trigram index would scale further but needs
  a sync story for older versions writing the file (no triggers: they would fail on SQLite builds without trigram).
  Not needed at this store's size; revisit if stores grow past ~50k facts.
- Paraphrase recall (no shared words): the optional online search by meaning (0.14.0; `fact_store search` beyond
  the words since 0.14.4). Offline, keywords stay the path (below).
- The word scan reads every row on each scan; past ~20k facts a persisted key index (with a reconciliation story for
  older versions writing the file) would be the next step. Host names in Spanish/French/Portuguese/Indonesian/German
  questions are not enforced as identifiers (compound words win there). German compounds in the question are not
  split (`Erdnussallergie`); keywords cover that today.
- Keywords for old facts: wired — hermes-dreaming's nightly job asks for them (`keywords_needed`, its cooldown and
  cap, main store only) and when it adds a fact; running on the maintainer's store.
- Search with no word in common (issue #3, finding 1 of the 0.14.4 reply): the fill keeps its bar; the measured
  variants (lower margin, top-1 above a z floor) answered more no-answer questions than they gained (CHANGELOG 0.14.5).
  Revisit with the user's dedicated negative set. Parallel searches mixing query vectors (finding 2) did not
  reproduce; `test_parallel_searches_never_share_a_query_vector` guards it.
- Loanwords (`_LOANWORDS` in `text.py`, 0.14.5): extend the list by exact spellings only; check every new Cyrillic
  spelling's forms (`_RU_ENDINGS`) against common words, never match by stem.
- Origin and confirmations (0.15.0): the nightly dreaming job should call `fact_feedback confirmed` for facts a
  person's message backs (it already finds them for promotion) — that, not repeated `add`, is where confirmations come
  from on a real store (0 repeats in 22 adds on the maintainer's). After a week or two, read `stats.origin` to see
  whether the agent fills `origin` at all; if most new facts are still `unknown`, the tool description needs work.
- Upstream PRs to the bundled copy that were never merged and are worth reviewing here: hermes-agent#133389
  (retrieval contract tests), #123599 (setup offers saved config).

Original authors are preserved in hermes-agent's history: `git log -- plugins/memory/holographic`.
