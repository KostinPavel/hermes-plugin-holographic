# Sanitized historical multilingual runs

The family examples in `provenance_multi_off.jsonl`, `provenance_multi_on.jsonl` and the ten `provenance_multi_m*_on.jsonl` margin-run files use fictional names, places and identifiers. Query texts and target keys are changed consistently. The original git_sha, numerical outcomes, measured timings, qid, endpoint and row order are retained as historical measurements. These files are not a fresh benchmark of the renamed corpus or of the current release; do not use them as evidence for either claim.

The unqualified files were sanitized in PR #39; the margin-run files are sanitized by the follow-up to PR #31. The current synthetic corpus is verified by `tests/test_holographic_languages.py`. Reproducing the full historical rerank/embedding comparison requires its model endpoints and a new labelled run. Git history has not been rewritten.
