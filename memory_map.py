"""Memory map: a read-only snapshot of ONE store as JSON, for a person to read (docs/memory-map.md).

The data layer of the map: a dashboard tab or a standalone page renders it; every privacy rule is applied here, so
the renderers cannot differ on them. The store is opened read-only (``mode=ro``, ``query_only``): nothing is written,
migrated or created, no model and no network are used. What the provider itself would not show is not shown:

- one store per export (one scope); the night's questions (dreaming) only when they are about this very store;
- a forgotten fact is its number and the date, nothing else (its text, keywords and names are gone from the store);
- a corrected version has no text (it never held); a text the provider's screen keeps out of answers (a credential,
  an instruction to the assistant, hidden markup) is not shown, current or past: the fact is listed under HIDDEN
  with its number and date.

    python -m <package>.memory_map --store PATH [--hermes-home DIR] [--tz ZONE] [--json]
    python <plugin dir>/memory_map.py --store PATH ...     (no Hermes core needed)

``export_map(store_path, hermes_home=None, tz=None, now=None)`` returns the same dict for a caller in Python."""

from __future__ import annotations

if __name__ == "__main__" and not __package__:  # run as a file: load the plugin's modules without its __init__
    import importlib
    import sys
    import types
    from pathlib import Path as _Path

    _here = str(_Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if _Path(p or ".").resolve() != _Path(_here)]  # its modules are not top-level
    _pkg = types.ModuleType("_holographic_map")
    _pkg.__path__ = [_here]
    sys.modules["_holographic_map"] = _pkg
    sys.exit(importlib.import_module("_holographic_map.memory_map").main())

import argparse
import json
import re
import sqlite3
import sys
import threading
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path

from .entity_names import fold
from .history import error_kind, source_of
from .retrieval import FactRetriever
from .screen import _recall_problem, _secret_in
from .store import MemoryStore

SCHEMA_VERSION = 1
# Words for trust (design, decided 29.10.2026), on the number rounded to hundredths (0.30000000000000004 is stored)
TRUST_NORMAL, TRUST_STRONG = 0.45, 0.65
STALE_DAYS = 60  # review()'s "never recalled" and "unconfirmed inference"
WEEK_DAYS = 7
_BIG = 100_000  # "all of them" for the provider's limits
_DREAM_FILES = {"proposals": ("cache", "dream-proposals.json"), "report": ("cache", "dream.json"),
                "queue": ("pending", "memory")}
_PROPOSAL_ID = re.compile(r"D-\d{1,9}")
_CODE = re.compile(r"[a-z_]{1,40}")
_FEED_ORDER = {"night": 0, "del": 1, "mrg": 2, "end": 3, "chg": 4, "conf": 5, "new": 6}
_REVIEW_REASONS = (("low trust", "low_trust"), ("never recalled", "never_recalled"),
                   ("an inference nobody confirmed", "unconfirmed_inference"), ("near-duplicate of #", "near_duplicate"))


class _ReadOnlyStore(MemoryStore):
    """The provider's store over a read-only connection of its own: the provider's read code (history, origins,
    entities, review, contradict) runs unchanged, and any write it might attempt fails instead of happening. Not in
    the shared connection registry, never migrated."""

    def __init__(self, db_path, tz: tzinfo | None = None) -> None:  # not MemoryStore's: that one opens for writing
        path = Path(db_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"no store at {path}")
        self.db_path, self.tz = path, tz
        self.default_trust, self.hrr_dim, self._hrr_available, self._key = 0.5, 1024, False, None
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False,
                               isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.create_function("hfold", 1, fold, deterministic=True)
            conn.execute("PRAGMA query_only = ON")
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        except BaseException:
            conn.close()
            raise
        self.tables = tables
        self._entry = {"conn": conn, "lock": threading.RLock(), "refs": 1, "ready": True, "migrated": False,
                       "generation": 0}
        self._conn, self._lock = conn, self._entry["lock"]

    def _meta(self, key: str) -> str | None:
        return super()._meta(key) if "meta" in self.tables else None  # before 0.4.0 there was no meta table

    def close(self) -> None:
        if self._entry is not None:
            self._entry = None
            self._conn.close()


# -- helpers --------------------------------------------------------------------------------------------------------

def _zone(tz) -> tuple[tzinfo, str]:
    """(tzinfo, its name): None = UTC; "+07:00" / "-03:30" a fixed offset; else an IANA name (zoneinfo)."""
    if tz is None or tz == "" or tz == "UTC":
        return timezone.utc, "UTC"
    if isinstance(tz, tzinfo):
        return tz, str(getattr(tz, "key", None) or tz)
    text = str(tz).strip()
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", text)
    if m:
        delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
        return timezone(-delta if m.group(1) == "-" else delta), text
    from zoneinfo import ZoneInfo
    return ZoneInfo(text), text


def _screen(text) -> str | None:
    """Why the provider would keep *text* out of an answer or away from a network call, as a code, or None: the
    strictest of its screens (recall's and the sendable one)."""
    value = str(text or "")
    problem = _recall_problem(value)
    if problem:
        return "instruction" if "instruction" in problem else "credential" if "credential" in problem else "markup"
    return "credential" if _secret_in(value) else None


def _shown(text):
    """*text* if it may be shown, else None."""
    return None if text is None or _screen(text) else text


def _round(value):
    return None if value is None else round(float(value), 2)


def _trust_level(trust) -> str | None:
    if trust is None:
        return None
    t = round(float(trust), 2)
    return "strong" if t >= TRUST_STRONG else "normal" if t >= TRUST_NORMAL else "weak"


def _day(store: _ReadOnlyStore, timestamp) -> str | None:
    """The local day of a UTC timestamp SQLite wrote, or None."""
    return store._local_day(timestamp) if timestamp else None


def _iso_day(value, zone: tzinfo) -> str | None:
    """The day of an ISO 8601 moment in *zone* (a date alone, or a moment without an offset, as written)."""
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        try:
            return date.fromisoformat(str(value)[:10]).isoformat()
        except ValueError:
            return None
    return (moment.astimezone(zone) if moment.tzinfo else moment).date().isoformat()


def _same_file(a, b) -> bool:
    try:
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return False


def _load_json(path: Path):
    """(state, data): state ok | absent | unreadable."""
    if not path.exists():
        return "absent", None
    try:
        return "ok", json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unreadable", None


# -- the snapshot ---------------------------------------------------------------------------------------------------

class _Snapshot:
    def __init__(self, store: _ReadOnlyStore, made: datetime, zone_name: str, hermes_home):
        self.s, self.made, self.zone_name, self.home = store, made, zone_name, hermes_home
        self.today = made.date().isoformat()
        self.week_from = (made.date() - timedelta(days=WEEK_DAYS)).isoformat()
        self.history = "fact_versions" in store.tables and store.history_ready()
        self.has = store.tables.__contains__

    def rows(self, sql, params=()):
        return [dict(r) for r in self.s.read(sql, params)]

    # -- facts --
    def build(self) -> dict:
        s = self.s
        current = {r["fact_id"]: r for r in self.rows(
            "SELECT fact_id, content, category, trust_score, retrieval_count, helpful_count, created_at, updated_at "
            "FROM facts ORDER BY fact_id")}
        versions: dict[int, list[dict]] = {}
        if self.history:
            for r in self.rows("SELECT * FROM fact_versions ORDER BY fact_id, seq"):
                versions.setdefault(r["fact_id"], []).append(r)
        merges = {r["from_fact_id"]: r for r in self.rows(
            "SELECT from_fact_id, into_fact_id, merged_at FROM fact_merges ORDER BY from_fact_id")} \
            if self.has("fact_merges") else {}
        forgotten = self.rows("SELECT fact_id, forgotten_at FROM forgotten ORDER BY fact_id") \
            if self.has("forgotten") and self.history else []
        gone = {r["fact_id"] for r in forgotten}
        # a fact that ended for good: no current version and not merged since (history_counts' rule)
        ended = {fid: [v for v in vs if v["status"] == "ended"][-1] for fid, vs in versions.items()
                 if fid not in current and fid not in gone and any(v["status"] == "ended" for v in vs)
                 and not any(v["status"] in ("current", "merged") for v in vs)}
        origins = s.origins(list(current))
        confirmations = self.rows("SELECT fact_id, confirmed_at, user_id FROM fact_confirmations "
                                  "ORDER BY confirmed_at, fact_id") if self.has("fact_confirmations") else []
        names = self._user_names(versions)

        hidden, facts = [], {}
        for fid, row in current.items():
            cur = next((v for v in versions.get(fid, []) if v["status"] == "current"), None)
            why = _screen(row["content"])
            if why:
                hidden.append({"id": fid, "d": cur["recorded_on"] if cur else _day(s, row["created_at"]),
                               "why": why, "status": "current"})
                continue
            facts[fid] = self._fact(fid, row["content"], row["category"], row["trust_score"], "current", cur,
                                    versions.get(fid, []), (origins.get(fid) or {}).get("origin"), row)
        for fid, last in sorted(ended.items()):
            why = _screen(last["content"])
            if why:
                hidden.append({"id": fid, "d": last["recorded_on"], "why": why, "status": "ended"})
                continue
            facts[fid] = self._fact(fid, last["content"], last["category"], last["trust_score"], "ended", last,
                                    versions[fid], last.get("src_origin"), None)
        hidden.sort(key=lambda h: h["id"])

        for c in confirmations:
            fact = facts.get(c["fact_id"])
            if fact is not None:
                fact["conf"].append({"d": _day(s, c["confirmed_at"]), "by": _shown(names.get(c["user_id"]))})
                if fact["origin"] not in ("said", "confirmed") and fact["status"] == "ended":
                    fact["origin"] = "confirmed"  # as origins() counts it for a current fact

        merged = []
        for from_id, m in merges.items():
            copy = [v for v in versions.get(from_id, []) if v["status"] == "merged"]
            if not copy:
                continue  # its versions are gone: the next writer drops the row
            text = None if _screen(copy[-1]["content"]) else copy[-1]["content"]
            entry = {"from": from_id, "into": m["into_fact_id"], "d": _day(s, m["merged_at"]), "prop": None,
                     "text": text}
            merged.append(entry)
            if m["into_fact_id"] in facts:
                facts[m["into_fact_id"]]["mergedFrom"].append({"id": from_id, "text": text, "d": entry["d"],
                                                               "prop": None})

        ents, links = self._entities(facts)
        for fid, fact in facts.items():
            fact["ents"] = links.get(fid, [])
        self._weak(facts)
        conflicts = self._conflicts(facts, links)
        dreaming, pending, nights = self._dreaming(facts, gone, versions, merged)
        pending = [{"kind": "conflict", "id": c["id"], "a": c["a"], "b": c["b"], "status": "open"}
                   for c in conflicts] + pending

        tombs = sorted(({"id": r["fact_id"], "d": _day(s, r["forgotten_at"])} for r in forgotten),
                       key=lambda t: (t["d"] or "", t["id"]), reverse=True)
        feed = self._feed(facts, versions, confirmations, names, forgotten, merged, nights)
        visible = list(facts.values())
        counts = {
            "current": sum(f["status"] == "current" for f in visible),
            "ended": sum(f["status"] == "ended" for f in visible),
            "merged": len(merged), "forgotten": len(forgotten),
            "corrected": sum(v["status"] == "corrected" for fid in facts for v in versions.get(fid, [])),
            "hidden": len(hidden), "entities": len(ents),
            "provisional": sum(e["standing"] == "provisional" for e in ents),
            "legacy": sum(f["status"] == "current" and f["via"] == "legacy" and f["origin"] == "unknown" for f in visible),
        }
        snap = {
            "made": self.made.isoformat(timespec="seconds"), "tz": self.zone_name, "store": s.db_path.name,
            "schema": {"store": s._meta("schema_version"), "history": s._meta("history_schema"),
                       "entity_rules": s._meta("entity_rules"), "experience": s._meta("experience_schema")},
            "counts": counts, "week_from": self.week_from,
            "excluded": [{"what": "hidden_by_screen", "count": len(hidden)}] if hidden else [],
            "experience": {"included": False}, "dreaming": dreaming,
        }
        return {"schema_version": SCHEMA_VERSION, "SNAP": snap, "ENT": ents, "FACTS": sorted(visible, key=lambda f: f["id"]),
                "CONFLICTS": conflicts, "PENDING": pending, "TOMBS": tombs, "MERGES": merged, "HIDDEN": hidden,
                "FEED": feed, "ATTEMPTS": []}

    def _user_names(self, versions) -> dict:
        """user_id -> the name the store recorded with it (the newest), to say who confirmed a fact."""
        out = {}
        for vs in versions.values():
            for v in vs:
                if v.get("src_user_id") and v.get("src_user_name"):
                    out[v["src_user_id"]] = (v["recorded_at"], v["src_user_name"])
        return {k: name for k, (_, name) in out.items()}

    def _fact(self, fid, text, category, trust, status, head, rows, origin, row) -> dict:
        s = self.s
        src = source_of(head) if head else None
        first = rows[0] if rows else None
        return {
            "id": fid, "text": text, "ents": [], "cat": _shown(category),
            "origin": origin if origin in ("said", "confirmed", "inferred") else "unknown",
            "trust": _round(trust), "trust_level": _trust_level(trust), "status": status,
            "from": head["valid_from"] if head else None,
            "to": head["valid_to"] if head and status == "ended" else None,
            "to_exact": bool(head["valid_to_exact"]) if head and status == "ended" else None,
            "rec": head["recorded_on"] if head else _day(s, row["created_at"]),
            "first": first["recorded_on"] if first else _day(s, row["created_at"]),
            "via": (head.get("src_via") or "legacy") if head else "legacy",
            "chat": _shown((src or {}).get("chat_type")), "by": _shown((src or {}).get("user_name")),
            "conf": [], "recalls": row["retrieval_count"] if row else None,
            "helpful": row["helpful_count"] if row else None, "last": None,
            "versions": [self._version(v) for v in rows if v["status"] != "merged"],
            "mergedFrom": [], "weak": [],
        }

    def _version(self, v) -> dict:
        out = {"seq": v["seq"], "status": v["status"], "text": None, "from": v["valid_from"], "to": v["valid_to"],
               "to_exact": bool(v["valid_to_exact"]), "rec": v["recorded_on"], "closed": _day(self.s, v["closed_at"]),
               "origin": v.get("src_origin") or None, "via": v.get("src_via") or None}
        if v["status"] != "corrected":  # a corrected version never held: never shown as the past
            if _screen(v["content"]):
                out["hidden"] = True
            else:
                out["text"] = v["content"]
        return out

    # -- entities --
    def _entities(self, facts) -> tuple[list[dict], dict[int, list[int]]]:
        s = self.s
        try:
            rows = s.list_entities(limit=_BIG)
        except sqlite3.OperationalError:  # a file older than entity_aliases (0.4.0): names only
            rows = [{"entity_id": r["entity_id"], "name": r["name"], "declared": r["entity_type"] == "declared",
                     "aliases": [], "text_aliases": [], "status": "established"}
                    for r in self.rows("SELECT entity_id, name, entity_type FROM entities")]
        since = {r["entity_id"]: r["created_at"] for r in self.rows("SELECT entity_id, created_at FROM entities")}
        stale = {r["entity_id"] for r in s.stale_provisional_entities(days=STALE_DAYS, limit=_BIG)} \
            if "entity_aliases" in s.tables else set()
        by_entity: dict[int, list[int]] = {}
        for r in self.rows("SELECT fact_id, entity_id FROM fact_entities ORDER BY entity_id, fact_id"):
            if r["fact_id"] in facts:
                by_entity.setdefault(r["entity_id"], []).append(r["fact_id"])
        out, links = [], {}
        for r in sorted(rows, key=lambda e: e["entity_id"]):
            eid = r["entity_id"]
            # a name only facts the screen hides gave, or a name the screen would hide itself, is not shown
            if not by_entity.get(eid) or _screen(r["name"]):
                continue
            out.append({"id": eid, "name": r["name"], "type": None, "declared": bool(r.get("declared")),
                        "aliases": [a for a in r.get("aliases", []) if _shown(a) is not None],
                        "text_aliases": [a for a in r.get("text_aliases", []) if _shown(a) is not None],
                        "standing": r.get("status") or "established", "stale": eid in stale,
                        "since": _day(s, since.get(eid)), "facts": len(by_entity[eid])})
            for fid in by_entity[eid]:
                links.setdefault(fid, []).append(eid)
        return out, links

    # -- weak basis, conflicts --
    def _weak(self, facts) -> None:
        """review()'s reasons for the current facts it lists, plus "no source" for a fact whose origin is unknown —
        except the records written before origins existed (via legacy): not weak for that (decided 29.10.2026)."""
        retriever = FactRetriever(self.s, hrr_weight=0.0)
        for item in retriever.review(limit=max(20, len(facts) + 1), stale_days=STALE_DAYS):
            fact = facts.get(item["fact_id"])
            if fact is None or fact["status"] != "current":
                continue
            code = next((c for prefix, c in _REVIEW_REASONS if item["reason"].startswith(prefix)), None)
            if code is None:
                continue
            fact["weak"].append(code)
            if code == "near_duplicate":
                of = int(item["reason"].rsplit("#", 1)[1])
                if of in facts:
                    fact["dup_of"] = of
        for fact in facts.values():
            if fact["status"] == "current" and fact["origin"] == "unknown" and fact["via"] != "legacy":
                fact["weak"].append("no_source")

    def _conflicts(self, facts, links) -> list[dict]:
        retriever = FactRetriever(self.s, hrr_weight=0.0)
        pairs = []
        for c in retriever.contradict(limit=_BIG):
            a, b = sorted((c["fact_a"]["fact_id"], c["fact_b"]["fact_id"]))
            if a in facts and b in facts:
                pairs.append({"a": a, "b": b, "score": c["contradiction_score"],
                              "shared": sorted(set(links.get(a, [])) & set(links.get(b, [])))})
        pairs.sort(key=lambda p: (-p["score"], p["a"], p["b"]))
        return [{"id": f"C{i}", **p} for i, p in enumerate(pairs, 1)]

    # -- dreaming --
    def _dreaming(self, facts, gone, versions, merged):
        """(SNAP.dreaming, pending items, night events). Only when the caller names the Hermes home, and only when
        the night's files are about this store: the profile store at <home>/memory_store.db, or the store the
        proposals name. A file that is missing, unreadable or of another shape is a state, never an error."""
        if self.home is None:
            return {"read": False, "reason": "not_requested", "files": {}}, [], []
        home = Path(self.home).expanduser()
        files, data = {}, {}
        for key, parts in _DREAM_FILES.items():
            path = home.joinpath(*parts)
            if key == "queue":
                files[key] = "absent" if not path.exists() else "ok" if path.is_dir() else "unreadable"
                continue
            files[key], data[key] = _load_json(path)
            if files[key] == "ok" and not isinstance(data[key], dict):
                files[key], data[key] = "unexpected", None
        proposals = (data.get("proposals") or {}).get("proposals")
        if files["proposals"] == "ok" and not isinstance(proposals, dict):
            files["proposals"], proposals = "unexpected", None
        proposals = {k: v for k, v in (proposals or {}).items() if _PROPOSAL_ID.fullmatch(str(k)) and isinstance(v, dict)}
        named = [v.get("store") for v in proposals.values() if v.get("store")]
        ours = _same_file(self.s.db_path, home / "memory_store.db") or any(_same_file(self.s.db_path, n) for n in named)
        if not ours:
            return {"read": False, "reason": "other_store", "files": files}, [], []
        current_version = {fid: v["version_id"] for fid, vs in versions.items() for v in vs if v["status"] == "current"}
        pending = self._proposals(proposals, facts, gone, current_version, merged)
        report = data.get("report") or {}
        nights = []
        day = _iso_day(report.get("generated_at"), self.made.tzinfo) if report.get("generated_at") else None
        if files["report"] == "ok":
            pending += self._report_items(report, facts, day)
            if day:
                stats = report.get("stats") if isinstance(report.get("stats"), dict) else {}
                alerts = report.get("alerts") if isinstance(report.get("alerts"), list) else []
                nights.append({"d": day, "k": "night", "at": str(report["generated_at"]),
                               "alerts": sorted(str(a.get("kind")) for a in alerts
                                                if isinstance(a, dict) and _CODE.fullmatch(str(a.get("kind")))),
                               "stats": {k: v for k, v in sorted(stats.items())
                                         if isinstance(v, (int, float)) and not isinstance(v, bool)}})
        if files["queue"] == "ok":
            queue = self._queue(home.joinpath(*_DREAM_FILES["queue"]))
            if queue:
                pending.append(queue)
        return {"read": True, "reason": None, "files": files}, pending, nights

    def _proposals(self, proposals, facts, gone, current_version, merged) -> list[dict]:
        out = []
        for pid in sorted(proposals, key=lambda k: int(k[2:])):
            rec = proposals[pid]
            if rec.get("store") and not _same_file(self.s.db_path, rec["store"]):
                continue  # about another store
            keep, copy = rec.get("keep") or {}, rec.get("copy") or {}
            ids = (keep.get("fact_id"), copy.get("fact_id"))
            if not all(isinstance(i, int) and not isinstance(i, bool) for i in ids) or set(ids) & gone:
                continue  # a question about a forgotten fact is gone with it
            if rec.get("status") == "done":
                for m in merged:
                    if (m["into"], m["from"]) == ids and m["prop"] is None:
                        m["prop"] = pid
                        if m["into"] in facts:
                            for f in facts[m["into"]]["mergedFrom"]:
                                if f["id"] == m["from"]:
                                    f["prop"] = pid
                continue
            if rec.get("status") != "open" or not all(i in facts for i in ids):
                continue  # declined / stale questions, or one about a fact the screen hides
            extra = copy.get("extra_words") if isinstance(copy.get("extra_words"), list) else []
            out.append({"kind": "merge", "id": pid, "keep": ids[0], "copy": ids[1], "covered": bool(copy.get("covered")),
                        "extra": [w for w in extra if isinstance(w, str) and _shown(w) is not None],
                        "created": _iso_day(rec.get("created"), self.made.tzinfo),
                        "asked": _iso_day(rec.get("last_published"), self.made.tzinfo),
                        "status": "open",
                        "current": all(current_version.get(i) == side.get("version_id")
                                       for i, side in zip(ids, (keep, copy)))})
        return out

    def _forgotten_text(self, text: str) -> bool:
        """Does *text* say what a forgotten fact said (the provider's tombstone check)?"""
        return bool(self.history and self.s._meta("tombstone_key") and self.s.forgotten_match(text))

    def _report_items(self, report, facts, day) -> list[dict]:
        out = []
        n = 0
        for item in report.get("conflicts") if isinstance(report.get("conflicts"), list) else []:
            if not isinstance(item, dict) or item.get("fact_id") not in facts:
                continue  # a fact forgotten (or hidden) since the night: nothing of it, its numbers included
            for hit in item.get("conflicts") if isinstance(item.get("conflicts"), list) else []:
                if not isinstance(hit, dict):
                    continue
                entry = hit.get("entry") if isinstance(hit.get("entry"), str) else None
                if hit.get("replace_unsafe") or entry is None or _screen(entry) or self._forgotten_text(entry):
                    continue
                n += 1
                out.append({"kind": "memory_conflict", "id": f"M-{n}", "fact": item["fact_id"], "entry": entry,
                            **{k: [str(x) for x in hit.get(k)] for k in ("fact_numbers", "memory_numbers",
                                                                          "fact_negations", "memory_negations")
                               if isinstance(hit.get(k), list)},
                            "asked": day, "status": "open"})
        for item in report.get("md_decays") if isinstance(report.get("md_decays"), list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("entry"), str):
                continue
            if _screen(item["entry"]) or self._forgotten_text(item["entry"]):
                continue
            days = item.get("last_mention_days")
            out.append({"kind": "relevant", "id": f"R-{item.get('index')}", "text": item["entry"],
                        "last_mention_days": days if isinstance(days, (int, float)) and not isinstance(days, bool)
                        else None, "asked": day, "status": "open"})
        return out

    def _queue(self, base: Path) -> dict | None:
        """Writes to long-term memory waiting for approval: how many and since when — never their text (they are
        not memory yet)."""
        records = writes = unreadable = 0
        oldest = None
        for path in sorted(base.glob("*.json")):
            state, data = _load_json(path)
            if state != "ok" or not isinstance(data, dict):
                unreadable += 1
                continue
            records += 1
            payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
            ops = payload.get("operations")
            writes += len([o for o in ops if isinstance(o, dict)]) if isinstance(ops, list) else 1
            created = data.get("created_at")
            if isinstance(created, (int, float)) and not isinstance(created, bool):
                oldest = created if oldest is None else min(oldest, created)
        if not records and not unreadable:
            return None
        day = datetime.fromtimestamp(oldest, self.made.tzinfo).date().isoformat() if oldest is not None else None
        return {"kind": "queue", "id": "Q", "records": records, "writes": writes, "unreadable": unreadable,
                "oldest": day, "status": "open"}

    # -- the week --
    def _feed(self, facts, versions, confirmations, names, forgotten, merged, nights) -> list[dict]:
        s, start, end = self.s, self.week_from, self.today
        inside = lambda d: d is not None and start <= d <= end  # noqa: E731
        out = []
        for fid, fact in facts.items():
            rows = versions.get(fid, [])
            if rows and inside(rows[0]["recorded_on"]):
                out.append({"d": rows[0]["recorded_on"], "k": "new", "f": fid})
            elif not rows and inside(fact["rec"]):
                out.append({"d": fact["rec"], "k": "new", "f": fid})
            by_id = {v["version_id"]: v for v in rows}
            for v in rows[1:]:
                if v["status"] in ("current", "changed", "ended") and inside(v["recorded_on"]):
                    before = by_id.get(v["supersedes"])
                    out.append({"d": v["recorded_on"], "k": "chg", "f": fid,
                                "how": "corrected" if before is not None and before["status"] == "corrected" else "changed"})
            if fact["status"] == "ended" and rows:
                closed = _day(s, rows[-1]["closed_at"]) if rows[-1]["status"] == "ended" else None
                if inside(closed):
                    out.append({"d": closed, "k": "end", "f": fid})
        for c in confirmations:
            d = _day(s, c["confirmed_at"])
            if c["fact_id"] in facts and inside(d):
                out.append({"d": d, "k": "conf", "f": c["fact_id"], "by": _shown(names.get(c["user_id"]))})
        for r in forgotten:
            d = _day(s, r["forgotten_at"])
            if inside(d):
                out.append({"d": d, "k": "del", "id": r["fact_id"]})
        for m in merged:
            if inside(m["d"]):
                out.append({"d": m["d"], "k": "mrg", "from": m["from"], "into": m["into"]})
        out += [n for n in nights if inside(n["d"])]
        key = lambda e: (e["d"], -_FEED_ORDER[e["k"]], -(e.get("f") or e.get("id") or e.get("from") or 0))  # noqa: E731
        return sorted(out, key=key, reverse=True)


def export_map(store_path, *, hermes_home=None, tz=None, now: datetime | None = None) -> dict:
    """The memory map of the store at *store_path* as a dict (schema_version, SNAP, ENT, FACTS, CONFLICTS, PENDING,
    TOMBS, MERGES, HIDDEN, FEED, ATTEMPTS). *hermes_home*: where the dreaming files are (none read without it).
    *tz*: the zone of the snapshot's dates (IANA name or "+07:00"; UTC by default). *now*: the snapshot's moment.
    Read-only; raises FileNotFoundError when there is no store file."""
    zone, zone_name = _zone(tz)
    made = (now or datetime.now(timezone.utc))
    made = (made if made.tzinfo else made.replace(tzinfo=timezone.utc)).astimezone(zone)
    store = _ReadOnlyStore(store_path, tz=zone)
    try:
        return _Snapshot(store, made, zone_name, hermes_home).build()
    finally:
        store.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only memory map of one holographic store, as JSON.")
    parser.add_argument("--store", required=True, help="the store file (memory_store.db or a scope's .db)")
    parser.add_argument("--hermes-home", help="read the dreaming files under this directory (only for its store)")
    parser.add_argument("--tz", help="zone of the dates: an IANA name or +HH:MM (default UTC)")
    parser.add_argument("--now", help="the snapshot's moment, ISO 8601 (default: now)")
    parser.add_argument("--json", action="store_true", help="print JSON (the only output format so far)")
    args = parser.parse_args(argv)
    try:
        now = datetime.fromisoformat(args.now) if args.now else None
        data = export_map(args.store, hermes_home=args.hermes_home, tz=args.tz, now=now)
    except FileNotFoundError as exc:
        print(f"memory_map: {exc}", file=sys.stderr)
        return 2
    except sqlite3.DatabaseError as exc:  # the kind only: SQLite may quote a stored value in its message
        print(f"memory_map: the store could not be read ({error_kind(exc)})", file=sys.stderr)
        return 1
    except ValueError as exc:  # --now or --tz
        print(f"memory_map: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
