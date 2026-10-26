"""tools/provenance_probe: an override never outlives its run, and close margins get their own file.

The probe writes three process globals — ``embeddings.RECALL_MARGIN``, ``ENDPOINT`` and ``SCALE``. An
override that survives its own run makes a later ``main([])`` write the earlier run's margin under the
plain file name and overwrite a baseline (review of PR #31, item 3); two margins closer than six
significant digits used to share one file name, the second overwriting the first (item 4). The runner is
stubbed: this is about the process, not about the model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import provenance_probe as pp  # noqa: E402


def _stub_run(langs, modes, emb, endpoint):
    """One row shaped like run_multilingual's — every field summarize() reads is here."""
    return [{"set": "multi", "lang": "en", "qid": "0", "split": "all", "kind": "real", "q": "query",
             "keys": ["f1"], "targets": ["t"], "hit": True, "hit5": True, "wrong": [], "wrong5": [],
             "sources": {"f1": ["words"]}, "miss_cls": "-", "mode": modes[0], "emb": emb, "out_n": 1}]


def _main(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(pp, "run_multilingual", _stub_run)
    return pp.main(["--set", "multi", "--modes", "off", "--out", str(tmp_path), *extra])


def _rows_in(path):
    """The rows a probe run wrote, one JSON object per line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_margin_override_does_not_outlive_its_run(monkeypatch, tmp_path):
    monkeypatch.setattr(pp._holographic_embeddings, "RECALL_MARGIN", 1.0)

    rows = _main(monkeypatch, tmp_path, "--recall-margin", "0.6")
    assert rows[0]["margin"] == 0.6  # stamped while the override was in force
    assert pp._holographic_embeddings.RECALL_MARGIN == 1.0  # and gone by the time main() returned

    again = _main(monkeypatch, tmp_path)  # the baseline run, same process
    assert again[0]["margin"] == 1.0
    plain = _rows_in(tmp_path / "provenance_multi_off.jsonl")
    assert plain[0]["margin"] == 1.0  # its own file: the 0.6 rows are not written over
    stamped = _rows_in(tmp_path / "provenance_multi_m0.6_off.jsonl")
    assert stamped[0]["margin"] == 0.6


def test_endpoint_and_scale_do_not_outlive_their_run(monkeypatch, tmp_path):
    # pinned on purpose: teardown undoes a leak, and the assertions below need a fixed starting point
    monkeypatch.setattr(pp, "SCALE", "logit")
    monkeypatch.setattr(pp, "ENDPOINT", pp.ENDPOINT)
    endpoint = pp.ENDPOINT

    first = _main(monkeypatch, tmp_path, "--endpoint", "http://a/v1/rerank", "--rerank-scale", "unit")
    assert first[0]["endpoint"] == "http://a/v1/rerank"
    assert first[0]["scale"] == "unit"
    assert (pp.ENDPOINT, pp.SCALE) == (endpoint, "logit")

    # no flags: --endpoint's default is the module's own value again, not the previous run's server
    second = _main(monkeypatch, tmp_path)
    assert second[0]["endpoint"] == (endpoint or "mock")
    assert second[0]["scale"] == "logit"


def test_two_close_margins_get_two_files(monkeypatch, tmp_path):
    """:g keeps six significant digits, so 0.6000001 and 0.6000002 both wrote _m0.6 and the second one
    silently overwrote the first; repr() round-trips the value (review of PR #31, item 4)."""
    _main(monkeypatch, tmp_path, "--recall-margin", "0.6000001")
    _main(monkeypatch, tmp_path, "--recall-margin", "0.6000002")
    files = sorted(p.name for p in tmp_path.glob("provenance_multi_m*_off.jsonl"))
    assert len(files) == 2, files

    # the published grid keeps the name it was committed under
    _main(monkeypatch, tmp_path, "--recall-margin", "0.6")
    assert (tmp_path / "provenance_multi_m0.6_off.jsonl").exists()