"""Fixtures for tools/mcnemar_gate.py (item 2 at 14:01 + the fix list at 19:07).

The partial-loss case stands fully paired on its own: if branch B ever reverts to the question-level
`c == 0`, gate_only (which keeps the question but drops one target) would pass and the first test would
fail. Unpaired real rows and unpaired chit rows get their own tests (both block both branches), and a
file that cannot be compared honestly — duplicates, mixed runs, targets disagreeing with off — must
raise instead of reporting."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "mcnemar_gate.py"
_spec = importlib.util.spec_from_file_location("mcnemar_gate", _TOOL)
assert _spec and _spec.loader
mcnemar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcnemar)


def _row(mode, qid, keys, targets, *, hit5=True, confused=False, git_sha=None):
    row = {"set": "household", "lang": "en", "qid": qid, "mode": mode, "kind": "real",
           "hit5": hit5, "keys": keys, "targets": targets, "confused": confused,
           "out_n": len(keys)}
    if git_sha:
        row["git_sha"] = git_sha
    return row


def _chit(mode, qid):
    # the same qid across all modes: chit pairing is checked just like real
    return {"lang": "en", "qid": qid, "mode": mode, "kind": "chit", "keys": [], "targets": [],
            "hit5": None, "confused": False, "out_n": 0}


def test_partial_loss_fully_paired(tmp_path, capsys):
    """Partial loss — a purely paired case: it blocks ONLY l5, there is no unpaired here (item 5 19:07)."""
    rows = [
        # q1: two targets; gate_only keeps hit5 (t1 in top-5) but t2 is pushed to the SIXTH position —
        # question-level c == 0, yet the per-target loss exists.
        _row("off", "q1", ["t1", "t2", "f1", "f2", "f3"], ["t1", "t2"], confused=True),
        _row("gate_only", "q1", ["t1", "f1", "f2", "f3", "f4", "t2"], ["t1", "t2"]),
        _row("gate_min2", "q1", ["t1", "t2", "f1", "f2", "f3"], ["t1", "t2"]),
        _chit("off", "c1"), _chit("gate_only", "c1"), _chit("gate_min2", "c1"),
    ]
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    mcnemar.report(str(path))
    out = capsys.readouterr().out

    gate_line = next(line for line in out.splitlines() if line.startswith("gate_only"))
    min2_line = next(line for line in out.splitlines() if line.startswith("gate_min2"))
    # the question is not lost (c == 0) but the target is lost — branch B is absent. If l5 reverted to c this test
    # would fail: pairing is complete, conf 0 < 1, neg are equal — B would appear.
    assert gate_line.split()[-1] == "-", gate_line
    # a control that the fixture can produce B at all (otherwise the test would only prove suppression)
    assert min2_line.split()[-1] == "B", min2_line
    assert "UNPAIRED" not in out


def test_unpaired_rows_fail_closed(tmp_path, capsys):
    """Complete pairing — otherwise B is never set, even when no losses are visible."""
    rows = [
        # all other branch B conditions hold: no losses (l5 = 0), confused reduced, neg equal —
        # the only remaining reason not to set B is incomplete pairing (UNPAIRED).
        _row("off", "q1", ["t1"], ["t1"], confused=True),
        _row("gate_only", "q1", ["t1"], ["t1"]),
        _row("gate_only", "q2", ["t2"], ["t2"]),  # only for gate_only
        _chit("off", "c1"), _chit("gate_only", "c1"),
    ]
    path = tmp_path / "unpaired.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    mcnemar.report(str(path))
    out = capsys.readouterr().out
    gate_line = next(line for line in out.splitlines() if line.startswith("gate_only"))
    assert gate_line.split()[-1] == "-", gate_line
    assert "UNPAIRED" in out


def test_unpaired_chit_rows_block(tmp_path, capsys):
    """Real is fully paired, gate_only has no chit block of its own — neg would be undercounted (item 5 19:07)."""
    rows = [
        _row("off", "q1", ["t1"], ["t1"], confused=True),
        _row("gate_only", "q1", ["t1"], ["t1"]),
        _chit("off", "c1"),  # the gate_only chit is missing
    ]
    path = tmp_path / "unpaired_chit.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    mcnemar.report(str(path))
    out = capsys.readouterr().out
    gate_line = next(line for line in out.splitlines() if line.startswith("gate_only"))
    assert gate_line.split()[-1] == "-", gate_line
    assert "UNPAIRED" in out


def test_duplicate_rows_fail(tmp_path):
    rows = [
        _row("off", "q1", ["t1"], ["t1"]),
        _row("off", "q1", ["t1"], ["t1"]),  # a duplicate: confused/lost would double up
        _row("gate_only", "q1", ["t1"], ["t1"]),
    ]
    path = tmp_path / "dup.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit):
        mcnemar.report(str(path))


def test_mixed_runs_fail(tmp_path):
    rows = [
        _row("off", "q1", ["t1"], ["t1"], git_sha="aaa1111"),
        _row("off", "q2", ["t2"], ["t2"], git_sha="bbb2222"),  # another run in the same file
        _row("gate_only", "q1", ["t1"], ["t1"], git_sha="aaa1111"),
    ]
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit):
        mcnemar.report(str(path))


def test_targets_mismatch_fail(tmp_path):
    rows = [
        _row("off", "q1", ["t1", "t2"], ["t1", "t2"]),
        _row("gate_only", "q1", ["t1", "t2"], ["t1"]),  # a different label: the loss «disappears»
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit):
        mcnemar.report(str(path))


def _a_rows(gate_confused: int):
    """12 questions: off hits 6, gate hits all 12 -> b=6, c=0, p=0.031, pts=+5.0."""
    rows = []
    for i in range(1, 13):
        hit = i > 6
        keys = [f"t{i}"] if hit else [f"f{i}"]
        rows.append(_row("off", f"q{i}", keys, [f"t{i}"], hit5=hit,
                         confused=i in (11, 12), git_sha="abc1234"))
        rows.append(_row("gate_only", f"q{i}", [f"t{i}"], [f"t{i}"], hit5=True,
                         confused=i in range(13 - gate_confused, 13), git_sha="abc1234"))
    rows.append(_chit("off", "c1"))
    rows.append(_chit("gate_only", "c1"))
    return rows


def test_branch_a_needs_confused_guard(tmp_path, capsys):
    """Unprotected A (item 4 19:07): +5 points at p<0.05 yields A only when conf/noise is not above off."""
    path = tmp_path / "a_clean.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _a_rows(gate_confused=1)),
                    encoding="utf-8")
    mcnemar.report(str(path))
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("gate_only"))
    assert "A" in line.split()[-1], line  # conf 1 <= 2, l5 = 0 -> A (and B)

    path = tmp_path / "a_guarded.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _a_rows(gate_confused=3)),
                    encoding="utf-8")
    mcnemar.report(str(path))
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("gate_only"))
    assert line.split()[-1] == "-", line  # conf 3 > 2 -> neither A nor B, although the points and p are the same


# --- --margin-pair: the gate on two recorded margins (review of PR #31) ---


def _mrow(margin, qid, *, keys, hit5, wrong=(), wrong5=(), targets=("t",), **over):
    """A row as provenance_probe stamps it for --recall-margin, in the multi shape by default: `wrong5`
    holds the confusable partners in the top five, `wrong` the same partners anywhere in the output.
    One clean head, so a pair is comparable on its own."""
    row = {"set": "multi", "lang": "en", "qid": qid, "margin": margin, "kind": "real", "q": "query",
           "split": "all", "mode": "off", "emb": "on", "scale": "logit", "git_sha": "958de48",
           "endpoint": "http://127.0.0.1:1237/v1/rerank", "targets": list(targets), "keys": list(keys),
           "hit5": hit5, "wrong": list(wrong), "wrong5": list(wrong5), "out_n": len(keys)}
    row.update(over)
    return row


def _pair(hi, lo, tmp_path):
    """One file per margin, as the probe writes them: hi = the current default (1.0), lo = the proposal."""
    default = tmp_path / "m1_on.jsonl"
    proposal = tmp_path / "m0.6_on.jsonl"
    default.write_text("\n".join(json.dumps(r) for r in hi), encoding="utf-8")
    proposal.write_text("\n".join(json.dumps(r) for r in lo), encoding="utf-8")
    return [str(default), str(proposal)]


def _rescued():
    """Six questions the proposal rescues (b=6, c=0, p=0.03125) plus one whose confusable partner X moves
    from position SIX — outside the top five — to position ONE: top-five confusion 0 -> 1 and nothing else
    the gate counts changes. This is the reviewer's reproduction of item 1."""
    hi, lo = [], []
    for i in range(6):
        hi.append(_mrow(1.0, f"r{i}", keys=["f1", "f2", "f3", "f4", "f5"], hit5=False))
        lo.append(_mrow(0.6, f"r{i}", keys=["t", "f1", "f2", "f3", "f4"], hit5=True))
    hi.append(_mrow(1.0, "c0", keys=["t", "a", "b", "c", "d", "X"], hit5=True, wrong=["X"]))
    lo.append(_mrow(0.6, "c0", keys=["X", "t", "a", "b", "c"], hit5=True, wrong=["X"], wrong5=["X"]))
    return hi, lo


def test_rising_confusion_blocks_the_gate(tmp_path, capsys):
    """Top-five confusion 0 -> 1 must fail the guard. Reading `wrong` as a fallback counted X on the default
    side too (1 -> 1), and the gate passed with six rescues at p = 0.031."""
    hi, lo = _rescued()
    mcnemar.report_margins(_pair(hi, lo, tmp_path))
    out = capsys.readouterr().out
    assert "PASS  pooled McNemar p < 0.05" in out  # every other guard is satisfied...
    assert "FAIL  confusion not higher (1 <= 0)" in out
    assert "gate DOES NOT PASS" in out


def test_an_equal_confusion_still_passes(tmp_path, capsys):
    """Control for the test above: the same six rescues with X left below the fifth position on both sides
    -> confusion 0 <= 0 and all four guards pass, so the fixture can pass and only the rise blocks it."""
    hi, lo = _rescued()
    lo[-1]["keys"] = ["t", "a", "b", "c", "d", "X"]
    lo[-1]["wrong5"] = []
    mcnemar.report_margins(_pair(hi, lo, tmp_path))
    out = capsys.readouterr().out
    assert "PASS  confusion not higher (0 <= 0)" in out
    assert "gate PASSES" in out


def test_two_endpoints_are_refused(tmp_path):
    """Two different servers are not one margin change: `endpoint` was never compared (item 2)."""
    hi, lo = _rescued()
    for r in hi:
        r["endpoint"] = "http://a/v1/rerank"
    for r in lo:
        r["endpoint"] = "http://b/v1/rerank"
    with pytest.raises(SystemExit, match="endpoint"):
        mcnemar.report_margins(_pair(hi, lo, tmp_path))


def test_a_run_without_an_endpoint_is_refused(tmp_path):
    """A field no row carries used to pass the mixed-run check: the filter dropped the empty values."""
    hi, lo = _rescued()
    for r in hi + lo:
        r.pop("endpoint")
    with pytest.raises(SystemExit, match="endpoint"):
        mcnemar.report_margins(_pair(hi, lo, tmp_path))


def test_a_pair_with_a_different_query_is_refused(tmp_path):
    hi, lo = _rescued()
    lo[-1]["q"] = "другой запрос"
    with pytest.raises(SystemExit, match="unpaired"):
        mcnemar.report_margins(_pair(hi, lo, tmp_path))


def test_a_pair_with_a_different_kind_is_refused(tmp_path):
    """real_keys came from the proposal alone, so a pair whose default row was another kind was reported
    as if both sides held the same question (item 2)."""
    hi, lo = _rescued()
    lo[-1]["kind"] = "chit"
    with pytest.raises(SystemExit, match="kind"):
        mcnemar.report_margins(_pair(hi, lo, tmp_path))


def test_a_pair_with_different_targets_is_refused(tmp_path):
    """The targets came from the default with a fallback to the proposal: a disagreeing pair lost and
    gained nothing because only one of the two lists was ever read."""
    hi, lo = _rescued()
    lo[-1]["targets"] = ["other"]
    with pytest.raises(SystemExit, match="targets"):
        mcnemar.report_margins(_pair(hi, lo, tmp_path))


def test_the_confusion_field_comes_from_the_rows_own_shape():
    """No fallback between the three shapes (item 1): multi's `wrong` covers the whole output."""
    assert not mcnemar._confused({"set": "multi", "wrong": ["x"], "wrong5": []})  # X below the fifth
    assert mcnemar._confused({"set": "multi", "wrong": ["x"], "wrong5": ["x"]})
    assert mcnemar._confused({"set": "household", "confused": ["x"]})
    assert not mcnemar._confused({"set": "household", "confused": []})
    assert mcnemar._confused({"set": "contrast_en", "wrong": True})
    assert not mcnemar._confused({"set": "contrast_en", "wrong": False})
    with pytest.raises(SystemExit, match="unknown row shape"):
        mcnemar._confused({"set": "multi"})


def test_a_multi_row_without_wrong5_cannot_pass_via_wrong(tmp_path):
    """An incomplete top-five measurement must be refused, never inferred from the whole output."""
    hi, lo = _rescued()
    hi[-1].pop("wrong5")
    with pytest.raises(SystemExit, match="wrong5"):
        mcnemar.report_margins(_pair(hi, lo, tmp_path))


@pytest.mark.parametrize("row,field", [
    ({"set": "multi", "wrong": ["x"]}, "wrong5"),
    ({"set": "household", "wrong5": ["x"]}, "confused"),
    ({"set": "contrast_en", "confused": ["x"]}, "wrong"),
])
def test_a_shape_requires_its_own_confusion_field(row, field):
    with pytest.raises(SystemExit, match=field):
        mcnemar._confused(row)


def test_other_shapes_fields_do_not_override_a_multi_measurement():
    assert not mcnemar._confused({"set": "multi", "wrong5": [], "confused": ["x"]})


def test_other_shapes_fields_do_not_hide_a_contrast_measurement():
    assert mcnemar._confused({"set": "contrast_en", "wrong": True, "wrong5": []})


def test_an_unknown_set_does_not_borrow_a_known_confusion_field():
    with pytest.raises(SystemExit, match="unknown row shape"):
        mcnemar._confused({"set": "unrecognized", "wrong": True})
