# Task outcomes and recorded memory explanations (opt-in)

Task usefulness is not fact truth. A correct fact about the wrong Alice must not lose trust just because a search
selected it for another person. This feature records retrieval attempts and their task outcomes separately from
`fact_feedback`; it never changes trust, origin, confirmations, fact versions or ranking.

In the `plugins.hermes-memory-store` section of `config.yaml`:

```yaml
experience: on
experience_trace: on  # optional: bounded candidate/stage ids; off still records the actual emission receipt
```

Both default to off. Restart the gateway / start a new session after configuration changes. With experience off,
the extra tool is absent and no new journal tables or receipts are created. Existing receipts remain subject to
forget triggers, even after collection is disabled. No new models, endpoints or dependencies are required.

Search returns `memory_attempt`; nonempty prefetch appends an opaque attempt id. Empty successful prefetch also
records a receipt, available through `fact_experience {action: recent}`. The id identifies an emission, not proof
that the model used that fact. A record contains exact fact/version references; a concurrent text change makes
recording fail safely rather than attach the wrong version. Failed collection does not suppress search results;
warnings and `recent.error_kind` expose it. Historical `as_of` search is currently not recorded.
When another process holds SQLite's write lock, recording an attempt uses a 50 ms SQLite busy timeout per lock
acquisition (or less if the connection already has a smaller timeout) and retries a busy failure once.
Each transaction is rolled back before retrying and current versions are checked again, so a stale or forgotten
fact cannot be attached as a current receipt. Actual elapsed time may be longer because of SQLite's retry intervals,
scheduling, other locks or disk delays. If all tries fail, search still returns without `memory_attempt`;
`recent.error_kind` and a warning report the failure. The attempt is not queued or recovered later, and a crash can
also prevent it being recorded. The connection's original busy timeout is restored for other operations. This is not
an overall search or disk-latency guarantee.
Attempts taking 25 ms or more write a warning with elapsed shared lock, schema, transaction begin, SQL work, commit
(or rollback), transaction and total time, try count and success or error status. Schema and transaction times sum
all tries; begin/work/commit describe the last try. A dash marks a stage not reached. SQLite
and operating-system scheduling can contribute to each phase; these timings don't prove a disk or locking cause.
The diagnostic includes no query, fact, correction, or identifier text.
The timing starts when journal recording begins, so a search already delayed before that point is outside this log.

The new tool is available only while collection is enabled:

```json
{"action":"outcome","attempt_id":"<receipt>","event_id":"task-17-result",
 "outcome":"corrected","reason":"wrong_entity","task":"Find the observatory engineer",
 "correction":"Use the engineer record for the other Alice.",
 "correction_refs":[{"fact_id":12,"version_id":23}]}
```

`correction_refs` is optional: when naming a known correct source, use its exact id and version from `history`.
The references belong to this store. `corrected` fixes the task result; it does not call `fact_store correct`.
An actual fact change still uses the existing change/correct protocol.

- Outcomes: `useful`, `dead_end`, `corrected`, `unknown`.
- Reasons: `wrong_entity`, `stale_version`, `irrelevant`, `missing_evidence`, `answer_error`, `other`.
- `task` is optional safe text, at most 160 characters; `correction` at most 400. Raw questions/answers are not
  collected. Existing safety screening applies. Do not copy unrelated private facts into either field.
- Delivery is idempotent by event_id. A conflicting payload under that id is refused. To revise an outcome,
  provide a new event_id and `supersedes` naming the active event. Stale concurrent revisions are refused.
- All events are explicitly **agent reports** (`human_verified: false`). The current core does not provide a
  trusted human-confirmation attestation for this tool. Reporting a human's correction does not authenticate it.
- One attempt has one active outcome; revisions preserve history. Different attempts are not automatically
  independent human evidence. Legacy helpful/unhelpful calls are not imported as extra evidence.
- Writes require the current store and original session. Read-only/subagent contexts cannot record outcomes.

`fact_experience {action: explain, attempt_id: "…"}` reads the original snapshot; it does not rerun retrieval or
call a model. It returns candidate stages where recorded, the exact emitted versions, current status/source
metadata for those versions, and outcome history. `recent` lists only the active session's receipts, in this store.
Missing/expired/unavailable information is labelled explicitly. `decisions` distinguishes recorded safety/repeat
rejections and observed coverage rejection; an unobserved exact cause remains `not_emitted_reason_not_recorded`.
Internal FTS scoring and every substage are not reconstructed. The default report does not include fact content.

The lazy additive schema is `experience_schema=1`. Limits per store: 30 days, 1000 attempts, 10 outcome revisions per
attempt, 16 stages with up to 100 candidate references each, 20 correction references. TTL/overflow pruning occurs
on the next receipt write; expired receipts cannot be read or receive new feedback. The journal is bounded, but
small fixed limits are not a performance guarantee for any machine or corpus.

Forgetting any referenced fact/version removes the entire dependent receipt, outcome text and correction references
in the same database transaction. SQL delete triggers also cover older writers that delete facts or versions.
The provider's existing forget scrub handles freed pages/WAL. Merge preserves history and is not forget; forgetting
a keep cleans its merged group as before. Late feedback cannot recreate a deleted receipt. External backups,
downloaded reports, Hermes session history and terminal logs are outside this journal's erase guarantee.

Dreaming reads this journal live, without a second text cache: see its `--memory-outcomes` / `--memory-explain`
commands. This first version helps diagnose memory use; it does not learn ranking or prove lower dialogue costs.
Disable collection to stop new records. Additive tables/triggers may remain when rolling back; do not remove the
erase triggers while retaining outcome text, and do not treat an old provider as an experience-journal reader.
