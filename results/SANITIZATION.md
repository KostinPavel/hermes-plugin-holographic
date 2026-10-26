# Sanitized historical multilingual runs

The family examples in `provenance_multi_off.jsonl` and `provenance_multi_on.jsonl` have been replaced with fictional names, places and identifiers. Their query keys were renamed consistently with the test fixtures. The original `git_sha`, measured timings and outcomes are retained as historical measurements. These files are not a fresh benchmark of the renamed corpus or of the current release; do not use them as evidence for either claim.

The current synthetic corpus is verified by `tests/test_holographic_languages.py`. Reproducing the full historical rerank/embedding comparison requires its model endpoints and a new labelled run. Git history has not been rewritten.
