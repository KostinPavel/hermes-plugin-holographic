# Sanitized historical multilingual margin runs

This change replaces private identifiers in the ten newly added `provenance_multi_m*_on.jsonl` margin-run files with the synthetic mapping used in main. Query texts and target keys are changed consistently. Every numerical outcome, timing, qid, row order, endpoint and original git_sha is retained: these remain historical measurements, not a fresh benchmark of the synthetic corpus or the current release.

The unqualified `provenance_multi_off.jsonl` and `provenance_multi_on.jsonl` files are not changed by this diff. Their sanitized versions are already in main through PR #39; this old-base branch must be integrated with main before publication. This note does not claim that those inherited local files have been sanitized here.

Reproducing the full rerank/embedding comparison requires its model endpoints and a new labelled run. Git history has not been rewritten.
