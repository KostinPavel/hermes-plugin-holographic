"""SQLite-backed fact store with entity resolution and trust scoring (single-user Hermes memory plugin)."""

import bisect
import contextlib
import itertools
from collections import Counter
import logging
import os
import re
import sqlite3
import threading
import unicodedata
from pathlib import Path

from . import holographic as hrr
from .embeddings import text_hash
from .history import HISTORY_SCHEMA, HistoryMixin, NotFound, parse_day
from .text import UNSEGMENTED, _script, analyze, is_greeting, is_identifier, name_keys, zh_phrase
from .text import fold as text_fold
from .text import STOPWORDS, is_chinese, ja_ko_phrase, zh_count, zh_word_cuts

logger = logging.getLogger(__name__)

# FTS is re-indexed only when indexed text changes: counters, trust and vectors are updated far more often.
_FACTS_AU = """CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF content, tags ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

""" + _FACTS_AU + """

-- 0.4.0. Older versions ignore these tables; see _migrate() for how their writes are reconciled on open.
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_key TEXT PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id),
    alias     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity ON entity_aliases(entity_id);

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
# 0.8.0 history tables (history.py) are created by a writer in _migrate(): a read-only context creates nothing.

# 0.14.0 search by meaning (embeddings.py): at most one vector per current fact, made by *model* from the text whose
# hash is *text_hash*. Created by every writer, whether or not the feature is on, so that deleting a fact always deletes
# its vector; older versions never read it.
_EMBEDDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id    INTEGER PRIMARY KEY,
    model      TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    vector     BLOB NOT NULL,
    revision   TEXT NOT NULL DEFAULT '',  -- random, new with every write: a vector made again is read again
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_VERSION = 2  # 1 = 0.1.x-0.3.x layout, 2 = + meta, entities.name_key, entity_aliases
# The rules entities are guessed from a fact's text by (meta.entity_rules; none recorded = 1, before 0.16.0). A store
# indexed under older rules is rebuilt on a writer's first open (rebuild_entity_index), up to the fact count below; a
# larger one on request. Raise it with every change of _extract_entities / _name_shaped that drops what they took.
_ENTITY_RULES = 4  # 2: 0.16.0; 3: 0.16.1 (script seams, line breaks, leading symbols, counts in quotes);
# 4: 0.16.2 (Chinese names that begin with a number, Chinese counts)
# Where an alias came from (entity_aliases.source, 0.16.0): 'user' (alias, rename, a merge someone asked for), 'aka'
# ("X aka Y" in a fact's text: it goes with the last fact that says it) or 'legacy' (written before 0.16.0 or by an
# older release since: kept like a given one, and dropped only when a forgotten fact said it as "X aka Y").
_GIVEN_ALIAS = "COALESCE(source, 'legacy') != 'aka'"
# What declared names split off aliases a fact's text gave during one operation (MemoryStore.collect_splits, 0.17.0):
# per thread, open only while the provider runs one tool call.
_SPLITS = threading.local()
_AUTO_REBUILD_MAX_FACTS = 2000
_DELETE_MAX = 100  # names in one delete_entity call
_AUTO_REBUILD_MAX_CHARS = 500_000  # and in all; the work grows with the text, not the count (review 0.16.0)
# 0.16.0: names delete_entity removed, as keyed hashes (see _name_digest). Created by a writer in _migrate().
_ENTITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_tombstones (
    digest     TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
_HELPFUL_DELTA, _UNHELPFUL_DELTA = 0.05, -0.10
_MAX_COUNTED_IDS = 500  # record_retrievals batch cap (far under SQLite's bound-parameter limit)

# Entity extraction, applied in order: capitalized multi-word phrases in any cased script ("John Doe", "Мария
# Иванова"), short quoted terms (straight, curly, guillemets; straight quotes only at word boundaries, so
# contractions like don't / it's never pair up), then both sides of "X aka Y".
# Combining marks are part of a word (0.16.1): \w leaves out the vowel signs of Indic and Thai script and Arabic
# harakat, so "श्री राम aka Ram" had no side. Ranges: combining diacritics, Arabic marks, the Indic blocks, Thai, Lao,
# Myanmar and Khmer signs.
_MARKS = "".join(f"{chr(a)}-{chr(b)}" for a, b in (
    (0x0300, 0x036F), (0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06ED), (0x0900, 0x0DFF),
    (0x0E31, 0x0E31), (0x0E34, 0x0E3A), (0x0E47, 0x0E4E), (0x0EB1, 0x0EB1), (0x0EB4, 0x0EBC), (0x0EC8, 0x0ECD),
    (0x102B, 0x103E), (0x17B4, 0x17D3)))
_RE_WORD = re.compile(r"(?:[^\W\d_]|[" + _MARKS + r"])+(?:-(?:[^\W\d_]|[" + _MARKS + r"])+)*")  # hyphenated: one word
_RE_TOKEN = re.compile(r"(?:\w|[" + _MARKS + r"])+(?:-(?:\w|[" + _MARKS + r"])+)*")  # aka sides: handles with digits
# Straight double quotes are found by _straight_quoted (0.16.1); single quotes keep \w beside them (don't, it's).
# Bounded by _QUOTED_MAX_CHARS (a longer span is no name anyway): an opening mark without a closing one is then
# read once, not to the end of the text from every opening mark (review 0.16.1: 5,000 "«" took 0.4 s).
_RE_QUOTED = tuple(re.compile(p.replace("+", "{1,60}")) for p in (
    r"(?<!\w)'([^'\n]+)'(?!\w)", r"«([^»\n]+)»", r"“([^”\n]+)”", r"„([^“”\n]+)[“”]",
    # 0.16.1: the corner and title marks of Chinese, Japanese and Korean (「华为」, 《平凡的世界》, 『東京大学』): in scripts
    # without case, quotes are the only way a name is found
    r"「([^」\n]+)」", r"『([^』\n]+)』", r"《([^》\n]+)》", r"〈([^〉\n]+)〉"))
# Spaces on one line (no line break of any kind: the str.splitlines boundaries) — also around "aka" (0.16.1)
_INLINE_SPACE_CHAR = "[^\\S\n\r\x0b\x0c\x1c\x1d\x1e\x85" + chr(0x2028) + chr(0x2029) + "]"
_INLINE_SPACE = _INLINE_SPACE_CHAR + "+"
# The word alone, with one inline space on each side as lookarounds: a leading run of spaces in the pattern made a
# text of 10,000 spaces quadratic (review 0.16.1); _aka_pairs checks the spaces around it.
_RE_AKA = re.compile(f"(?<={_INLINE_SPACE_CHAR})(?:aka|a\\.k\\.a\\.|also{_INLINE_SPACE}known{_INLINE_SPACE}as)"
                     f"(?={_INLINE_SPACE_CHAR})", re.IGNORECASE)
# A comma too (0.16.0): the legacy aliases column is comma-separated, see _legacy_aliases.
_RE_NOT_A_NAME = re.compile(r"[=/\\@<>{}\[\]|`,]|://|\b\d{1,3}(?:\.\d{1,3}){3}\b")
# A quoted span longer than this is a quotation ("she said: 'see you on Friday'"), not a name. Chinese and Japanese
# write a quotation without spaces: more Han characters than _QUOTED_MAX_HAN is one too (中国人民解放军总医院 has 10).
_QUOTED_MAX_WORDS, _QUOTED_MAX_CHARS, _QUOTED_MAX_HAN = 3, 60, 16
_RE_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_RE_VERSION = re.compile(r"[vV]?\d+(?:\.\d+)+")  # 0.17.0, v2.1: a release, not a name (a whole word; gpt-3.5-turbo stays)
_PHRASE_MAX_WORDS, _ENTITY_MAX_CHARS = 5, 60
_AKA_MAX, _AKA_WINDOW = 16, 4 * _ENTITY_MAX_CHARS  # "X aka Y" read per text, and how far from the "aka" (0.16.0)
# Capitalized only because they open a sentence: dropped from the front of a capitalized run ("So Apple").
_SENTENCE_STARTERS = frozenset("""
    a an and as at but by for from here if in it its my no not now of on or our so that the then there these this
    those to today tomorrow we when where yes yesterday you your
    а в во вот вчера да если завтра и или когда мой моя мои на наш наша наши не но однако по при сегодня так там
    тут у это этот эта эти""".split())
_ENTITY_NAMES_SQL = "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?"


class DuplicateContentError(ValueError):
    """update_fact would make a fact identical to another one (content is UNIQUE)."""

    def __init__(self, fact_id: int):
        super().__init__(f"another fact already has this content (fact_id {fact_id})")
        self.fact_id = fact_id


def _clamp_trust(value: float) -> float:
    return max(0.0, min(1.0, value))


def fold(text):
    """Comparison key for names: Unicode casefold with whitespace collapsed. SQLite's lower()/LIKE fold ASCII only,
    so this is registered on the connection as hfold() for entity lookups."""
    return " ".join(text.casefold().split()) if isinstance(text, str) else text


def entity_key(name: str) -> str:
    """Identity key of an entity name: NFKC, casefold, hyphens/underscores as spaces, whitespace collapsed, edge
    punctuation stripped. "Hermes Agent", "hermes-agent" and "HERMES  AGENT" are one entity; "coop" and "co-op" are
    not merged into each other (a hyphen becomes a space, it is not deleted)."""
    text = " ".join(unicodedata.normalize("NFKC", name or "").casefold().replace("-", " ").replace("_", " ")
                    .replace("ё", "е").split())  # Семён is Семен (0.16.1)
    text = text.strip(_KEY_EDGES).rstrip(_KEY_EDGES + ".")
    # A leading dot before a letter or a digit is part of the name (0.16.1): ".NET" is not "net", whose key it had,
    # so the word "net" in "a fishing net" mentioned .NET
    return text if text[:1] == "." and text[1:2].isalnum() else text.lstrip(_KEY_EDGES + ".")


_KEY_EDGES = " ,;:!?'\"«»“”„‘’"  # ‘ ’ since 0.16.1


def _entity_key_0160(name: str) -> str:
    """entity_key as 0.16.0 and earlier computed it: the key a tombstone of then was written under, and that older
    rows hold until _rekey_entities moves them."""
    text = unicodedata.normalize("NFKC", name or "").casefold().replace("-", " ").replace("_", " ")
    return " ".join(text.split()).strip(" .,;:!?'\"«»“”„")


def _is_capitalized(word: str) -> bool:
    """'John', 'Мария', 'Jean-Luc': each hyphen part an upper-case letter followed by lower-case ones (so not
    'NASA', 'I' or 'iPhone')."""
    return all(len(part) > 1 and part[0].isupper() and part[1:].islower() for part in word.split("-"))


class _Piece:
    """A part of a regex match, with the match's start / end / group interface."""
    __slots__ = ("_start", "_end", "_text", "word")

    def __init__(self, start: int, end: int, text: str, word: tuple[int, int]):
        self._start, self._end, self._text = start, end, text
        self.word = word  # (start, end) of the whole match the piece came from

    def start(self) -> int:
        return self._start

    def end(self) -> int:
        return self._end

    def group(self, _=0) -> str:
        return self._text


def _script_pieces(text: str, pattern: re.Pattern) -> list[_Piece]:
    """The matches of *pattern* in *text*, each split where its letters change script or letter case (Latin, Cyrillic,
    Greek; Han, kana, Hangul, Thai...): "Hub直连可用" is two pieces, "Hub" and "直连可用", and "Hubе" two (0.16.1, issue #4:
    "Docker Hub直连可用" was taken as one name). Hyphens and digits go with the letters after them; two pieces of one word
    have no gap between them, so a run of names never crosses the seam."""
    out = []
    for m in pattern.finditer(text):
        word, base = m.group(0), m.start()
        start, key, last = 0, None, -1
        for i, c in enumerate(word):
            if not c.isalpha() or unicodedata.category(c) == "Lm":  # ʻ ʼ ー 々 hold a word together (0.16.1)
                continue
            cased = c.lower() != c.upper()
            k = (_script(c), cased, "" if cased else _uncased_block(c))
            if key is not None and k != key:
                out.append(_Piece(base + start, base + last + 1, word[start:last + 1], (base, m.end())))
                start = last + 1
            key, last = k, i
        out.append(_Piece(base + start, base + len(word), word[start:], (base, m.end())))
    return out


def _hugs(text: str, i: int) -> bool:
    """Whether text[i] is a digit or a letter with case: a straight double quote beside one is no quote (a 5" screen,
    an inch mark inside a word). A letter of a script without case may touch it: Chinese writes 用"Docker Hub"直连."""
    if not 0 <= i < len(text):
        return False
    c = text[i]
    return c.isdigit() or (c.isalpha() and c.lower() != c.upper())


def _straight_quoted(text: str) -> list[str]:
    """Spans between straight double quotes on one line, as a regex with lookarounds would pair them: an opening quote
    not right after a digit or a letter with case, the next quote closing it unless a digit or such a letter follows
    (0.16.1: the old word-character lookarounds missed 用"Docker Hub"直连, and character ranges missed full-width letters)."""
    return [text[i + 1:j - 1] for i, j in _straight_quote_spans(text)]


def _straight_quote_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each span _straight_quoted finds, its quotes included."""
    quotes = [i for i, c in enumerate(text) if c == '"']
    out, k = [], 0
    while k < len(quotes) - 1:
        i, j = quotes[k], quotes[k + 1]
        if not _hugs(text, i - 1) and i + 1 < j <= i + 1 + _QUOTED_MAX_CHARS and "\n" not in text[i + 1:j] \
                and not _hugs(text, j + 1):
            out.append((i, j + 1))
            k += 2
        else:
            k += 1
    return out


# Quotes that hold what someone said, a label or a word in passing as often as a name — the title marks 《》〈〉 hold
# titles and are not among them (0.16.2). An opening mark and its closing ones.
_PLAIN_QUOTE_PAIRS = {"«": "»", "“": "”", "„": "“”", "「": "」", "『": "』", '"': '"', "'": "'", "‘": "’"}
_APOSTROPHES = "'’"  # quotes, and apostrophes inside a word: don't, O'Brien, don’t
_RE_LINE_BREAK = re.compile("[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")


def _word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _cased_word_in(text: str, word: str) -> bool:
    """Whether *word* (letters with case) stands in *text* as a word: no digit or letter with case right beside it, so
    a word glued to Chinese counts ("新的huawei手机": the seam of _script_pieces), "huaweis" does not."""
    def joins(c: str) -> bool:
        return c.isdigit() or c == "_" or (c.isalpha() and c.lower() != c.upper())
    i = text.find(word) if word else -1
    while i >= 0:
        j = i + len(word)
        if not (i > 0 and joins(text[i - 1])) and not (j < len(text) and joins(text[j])):
            return True
        i = text.find(word, i + 1)
    return False


def _uncased(name: str) -> bool:
    """A name with letters and none of them with case: Chinese, Japanese, Thai, Arabic... (0.16.2). Text finds such a
    name only in quotes (or in "X aka Y"): it has no capitals to stand out by. One Latin letter gives it case ("变量x"):
    such a name keeps the two-fact rule of the names with capitals."""
    letters = [c for c in name if c.isalpha()]
    return bool(letters) and all(c.lower() == c.upper() for c in letters)


def _outside_plain_quotes(text: str) -> str:
    """*text* with code and everything in plain quotes blanked (line breaks, so no name runs across), read in NFKC as
    mentions are ("＂" is a quote too); the title marks stay (0.16.2). Stricter than the extractor on purpose — a quote
    of any length, quotes nested in one of their own kind («a «b» c»), a quote never closed (to the end of its line),
    single quotes (but not an apostrophe inside a word: don't, O'Brien) — since what is blanked here only keeps a name
    provisional. Each character is looked at once or twice. Not cached: the store keeps what it needs per fact
    (_used_outside_quotes), and a process-wide cache would keep forgotten text in memory (review 0.16.2)."""
    text = unicodedata.normalize("NFKC", _without_code(unicodedata.normalize("NFC", text or "")))
    chars, i, n = list(text), 0, len(text)
    ends = [m.start() for m in _RE_LINE_BREAK.finditer(text)] + [n]  # any line break (str.splitlines)
    while i < n:
        opener = text[i]
        closers = _PLAIN_QUOTE_PAIRS.get(opener)
        if closers is None or (opener in _APOSTROPHES and i > 0 and _word_char(text[i - 1])):
            i += 1
            continue
        eol = ends[bisect.bisect_right(ends, i)]  # the end of its line
        depth, close, k = 1, eol - 1, i + 1
        while k < eol:
            c = text[k]
            if c in closers and not (c in _APOSTROPHES and k + 1 < n and _word_char(text[k + 1])):
                depth -= 1
                if depth == 0:
                    close = k
                    break
            elif c == opener and opener not in closers:
                depth += 1
            k += 1
        chars[i:close + 1] = "\n" * (close + 1 - i)
        i = close + 1
    return "".join(chars)


_RE_FENCE = re.compile(r"[ ]{0,3}(`{3,}|~{3,})")
_RE_TICKS = re.compile(r"`+")


def _without_code(text: str) -> str:
    """*text* with Markdown code blanked out, offsets kept: code is not a name (0.16.1: `Alice Smith` in a code span
    was a person, and "True aka False" a pair). Fenced blocks — ``` with no backtick after it on the line (else it is
    an inline span), to the closing fence or the end; ~~~ only with a closing fence, since a line of tildes is as often
    a rule — and inline spans of any run of backticks closed by a run of the same length; one pass. Indented lines are
    not taken for code: in notes they are nested lists ("Family:⏎    - Alice Smith", review 0.16.1). Blanked with NUL,
    neither a letter nor a space, so no name runs across it."""
    if "`" not in text and "~~~" not in text:
        return text
    chars = list(text)
    lines = text.splitlines(keepends=True)
    fences = [_RE_FENCE.match(line) for line in lines]
    # the last line that is a ~~~ fence: one opens a block only if another comes after it (review 0.16.2: looking
    # at every later line from every fence took 23 s on 80,000 lines of "~~~")
    last_tilde = max((n for n, f in enumerate(fences) if f is not None and f.group(1)[0] == "~"), default=-1)
    pos, fence = 0, None
    for n, (line, m) in enumerate(zip(lines, fences)):
        if m is not None and fence is None:
            mark = m.group(1)[0]
            opens = "`" not in line[m.end():] if mark == "`" else last_tilde > n
            if opens:
                fence = mark * 3
                chars[pos:pos + len(line)] = "\0" * len(line)
        elif fence is not None:
            chars[pos:pos + len(line)] = "\0" * len(line)
            if m is not None and m.group(1).startswith(fence):
                fence = None
        pos += len(line)
    text = "".join(chars)
    opened = None
    for m in _RE_TICKS.finditer(text):
        if opened is None:
            opened = m
        elif len(m.group(0)) == len(opened.group(0)):
            chars[opened.start():m.end()] = "\0" * (m.end() - opened.start())
            opened = None
    return "".join(chars)


def _mixed(name: str) -> bool:
    """A word of two scripts or cases glued together ("华为mate60"): read as strictly as a name in quotes."""
    return any(p.word != (p.start(), p.end()) for p in _script_pieces(name, _RE_TOKEN))


_ZERO_WIDTH_JOINERS = chr(0x200C) + chr(0x200D)
_MENTION_KEEPS = frozenset("'’&-‐‑_@#*%+")  # inside a word or a name: o'brien, at&t, jean-luc, c++


def _mention_stretches(text: str) -> list[str]:
    """*text* cut at everything the words of one name never stand across (0.16.1): line breaks and control characters,
    invisible format characters, brackets, quotes and other punctuation but the few that hold words together
    (_MENTION_KEEPS), and emoji. The analysis keeps the words and drops what stood between them, so before this
    "Alice\nSmith", "Alice, Smith" and two items of a list mentioned Alice Smith."""
    out, start = [], 0
    for i, c in enumerate(text):
        if c == "\t":
            continue
        cat = unicodedata.category(c)
        if cat in ("Cc", "Zl", "Zp", "So", "Sk") or (cat == "Cf" and c not in _ZERO_WIDTH_JOINERS) \
                or (cat[0] == "P" and c not in _MENTION_KEEPS and not _holds_word(text, i)):
            if i > start:
                out.append(text[start:i])
            start = i + 1
    if start < len(text):
        out.append(text[start:])
    return [_split_seams(stretch) for stretch in out]


def _holds_word(text: str, i: int) -> bool:
    """Whether the punctuation at *i* is part of a name, not a cut (review 0.16.1): a dot, colon or slash between two
    letters or digits (Node.js, Яндекс.Такси, 1С:Предприятие, app.example.com), or the dot after an initial or a short
    abbreviation (J. R. R. Tolkien, А. С. Пушкин, Dr. Alice Smith, U.S. Army). "Alice. Smith went" stays cut."""
    c = text[i]
    if c in ".:/" and 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum():
        return True
    if c != ".":
        return False
    j = i
    while j > 0 and text[j - 1].isalpha():
        j -= 1
    return 1 <= i - j <= 2 and (j == 0 or not text[j - 1].isalnum())


def _split_seams(stretch: str) -> str:
    """*stretch* with a space where a word changes script, as the extractor reads it (0.16.1): "用Docker Hub直连" and
    "на Docker Hubе" mention Docker Hub."""
    cuts = [piece.start() for piece in _script_pieces(stretch, _RE_TOKEN) if piece.start() != piece.word[0]]
    if not cuts:
        return stretch
    out, last = [], 0
    for cut in cuts:
        out += (stretch[last:cut], " ")
        last = cut
    return "".join(out) + stretch[last:]


def _found_between_cuts(folded: str, name: str, cuts: frozenset) -> bool:
    """Whether *name* occurs in *folded* Chinese text beginning and ending at word cuts (or beside a character that is
    not Han, where a Han word ends anyway)."""
    def cut(i: int) -> bool:
        return i in cuts or i == 0 or i == len(folded) or not (
            _RE_HAN.match(folded[i - 1]) and _RE_HAN.match(folded[i]))
    start = folded.find(name) if name else -1
    while start >= 0:
        if cut(start) and cut(start + len(name)):
            return True
        start = folded.find(name, start + 1)
    return False


def _uncased_block(c: str) -> str:
    """Which writing a letter without case belongs to, for _script_pieces: Hangul, Thai, Lao, Khmer, Myanmar apart, and
    Han with kana as one (a Japanese word mixes them: 山田さん). Review 0.16.1: "测试삼성" was one piece."""
    name = unicodedata.name(c, "")
    for block in ("HANGUL", "THAI", "LAO", "KHMER", "MYANMAR"):
        if name.startswith(block):
            return block
    return "CJK" if UNSEGMENTED.match(c) else _script(c)


def _inline_gap(gap: str) -> bool:
    """Spaces between two words of one name: some, and no line break of any kind — str.splitlines' boundaries: LF, CR,
    VT, FF, NEL, the Unicode line and paragraph separators (0.16.1: a heading and a name on the next line were taken
    as one name)."""
    return bool(gap) and gap.isspace() and len(("x" + gap + "x").splitlines()) == 1 \
        and not any(unicodedata.category(c) == "Cc" for c in gap)  # a tab too: cells of a table (review 0.16.1)


def _capitalized_phrases(text: str) -> list[str]:
    """Names of two to five capitalized words in any cased script, minus a sentence-opening function word."""
    phrases = []
    words_in_order = _script_pieces(text, _RE_WORD)
    run: list[_Piece] = []

    def flush():
        words = run[1:] if run and fold(run[0].group(0)) in _SENTENCE_STARTERS else run
        if 2 <= len(words) <= _PHRASE_MAX_WORDS:
            phrases.append(text[words[0].start():words[-1].end()])

    for m in words_in_order:
        if _is_capitalized(m.group(0)) and (not run or _inline_gap(text[run[-1].end():m.start()])):
            run.append(m)
            continue
        flush()
        run = [m] if _is_capitalized(m.group(0)) else []
    flush()
    return phrases


def _aka_side(segment: str, *, from_end: bool) -> str | None:
    """The name right before (from_end) or right after an 'aka': a capitalized run, else a single token."""
    tokens = _script_pieces(segment, _RE_TOKEN)
    if not tokens:
        return None
    edge = tokens[-1] if from_end else tokens[0]
    if (edge.end() != len(segment)) if from_end else (edge.start() != 0):
        return None  # punctuation between the name and 'aka'
    if edge.word != (edge.start(), edge.end()):
        # a word of two scripts ("华为mate60 aka Huawei"): the whole word is the side, alone, and _aka_pairs judges it
        # as a name in quotes is judged (0.16.1: cutting it at the seam lost a model name and its alias)
        return segment[edge.word[0]:edge.word[1]]
    picked = [edge]
    if _is_capitalized(edge.group(0)):
        neighbours = reversed(tokens[:-1]) if from_end else tokens[1:]
        for tok in neighbours:
            gap = segment[tok.end():picked[-1].start()] if from_end else segment[picked[-1].end():tok.start()]
            if len(picked) >= _PHRASE_MAX_WORDS or not _inline_gap(gap) or not _is_capitalized(tok.group(0)):
                break
            picked.append(tok)
    if from_end:
        picked.reverse()
    return segment[picked[0].start():picked[-1].end()]


def _clean_given(name) -> str:
    """A name someone gives, without what they cannot have meant: edge punctuation (a leading dot stays: ".NET") and
    invisible format characters such as the bidi marks a browser leaves in an Arabic or Hebrew name (ZWJ / ZWNJ stay
    where the script writes them). Review 0.16.1: such a name was refused, silently."""
    if not isinstance(name, str):
        return ""
    name = "".join(c for i, c in enumerate(name) if unicodedata.category(c) != "Cf" or _joiner_ok(name, i))
    return name.strip(" ,;:!?").rstrip(".")


def _legacy_aliases(names) -> str:
    """The legacy comma-separated entities.aliases value, which older versions resolve aliases from and every writer
    open imports again by splitting it on commas: a name with a comma is left out of it (entity_aliases keeps it whole),
    or "Smith, John" would come back as the aliases "Smith" and "John" (review 0.16.0)."""
    return ",".join(dict.fromkeys(n.strip() for n in names if isinstance(n, str) and n.strip() and "," not in n))


def _valid_entity(name: str, *, given: bool = False) -> bool:
    """Reject what is not a name: too short or long, not beginning with a letter or a digit (".NET" may: a dot and a
    capital), key=value, paths, URLs, IPs, mostly non-letters. The
    check for names someone gave (entities=[...], alias, rename); names guessed from text pass _name_shaped too."""
    if not 2 <= len(name) <= _ENTITY_MAX_CHARS or _RE_NOT_A_NAME.search(name):
        return False
    # The same checks on the form names are compared in (0.16.1): "https：／／example.com" is a URL once NFKC folds its
    # full-width colon and slashes, as entity_key does; and no invisible format character (a zero-width space makes a
    # copy of a deleted name that its tombstone does not know) — but ZWJ / ZWNJ, which some scripts write words with.
    folded = unicodedata.normalize("NFKC", name)
    if _RE_NOT_A_NAME.search(folded) or len(name.splitlines()) > 1 or any(
            unicodedata.category(c) == "Cc" or (unicodedata.category(c) == "Cf" and not _joiner_ok(name, i))
            for i, c in enumerate(name)):
        return False
    if not (name[0].isalnum() or (name[0] == "." and name[1:2].isupper())  # 0.16.1: "+.siliconflow.cn", "-flag"
            or (given and name[0] in ".'’$" and name[1:2].isalnum())):  # given: ".net", "'s-Hertogenbosch", "$GME"
        return False
    visible = [c for c in name if not c.isspace()]
    if given:  # someone named it: "3M", "O2", "C++" are names; a letter is enough
        return any(c.isalpha() for c in visible)
    if name[0].isdigit() and is_chinese(name):
        # 0.16.2 (issue #4): a Han character is a word, so the share of letters below dropped half the Chinese names
        # that begin with a number (360安全卫士, 115网盘, 36氪) and kept some counts ("3个月" yes, "15分钟" no). A
        # number and units is a count (text.zh_count); anything else after the number makes a name.
        return not zh_count(name) and any(c.isalpha() for c in visible)
    # vowel signs and other marks count with their letters (review 0.16.1: "श्री" fell under the bar)
    return sum(c.isalpha() or unicodedata.category(c)[0] == "M" for c in visible) >= 0.6 * len(visible)


_JOINER_SCRIPTS = frozenset({"arabic", "devanagari", "bengali", "indic"})


def _joiner_ok(name: str, i: int) -> bool:
    """A zero-width joiner or non-joiner between two letters (or a letter and a mark) of a script that writes words
    with them; anywhere else it only makes a copy of a name that its key and tombstone do not know (review 0.16.1:
    "Alice\u200d Smith" came back after "Alice Smith" was deleted)."""
    if name[i] not in "\u200c\u200d" or not 0 < i < len(name) - 1:
        return False
    return all(_script(c) in _JOINER_SCRIPTS for c in (name[i - 1], name[i + 1]) if c.isalpha()) and \
        any(c.isalpha() or unicodedata.category(c)[0] == "M" for c in (name[i - 1], name[i + 1]))


_NOT_A_NAME = (f"not a name (2-{_ENTITY_MAX_CHARS} characters, mostly letters; no paths, URLs, key=value, commas, "
               "line breaks or flags)")


def unusable_names(names) -> list[dict]:
    """The names in *names* that a declaration (entities=[...]) cannot link, with why: what _link_declared skips. A
    fact is stored all the same; the agent is told which names did not link, rather than believing they did."""
    out = []
    for name in names or ():
        if not isinstance(name, str):
            out.append({"name": repr(name)[:80], "reason": "not a string"})
        elif not _valid_entity(_clean_given(name), given=True):
            out.append({"name": name[:80], "reason": _NOT_A_NAME})
    return out


def _name_shaped(name: str, *, quoted: bool) -> bool:
    """_valid_entity, and for a name guessed from a fact's text (0.16.0) also not: a version number with nothing that
    names a thing beside it ("0.17.0 installed", «версия 0.17.0»; "Python 3.12" and «1С:Предприятие 8.3» stay, for a
    word with a capital or in an uncased script names one); in quotes, a span in a cased script all in lower case with
    no identifier in it ('count', 'run fast', «быстро беги»; 'kanban.db', 'gpt-4o' and 'hr-ovh' stay), a greeting or
    courtesy ("Good morning", «Спасибо большое», “谢谢”: text.is_greeting), a Chinese span that reads as something said
    (text.zh_phrase: “先去搜索再告诉我答案”, “知道了”; 平凡的世界, 有赞科技有限公司 and 饿了么 stay), or more Han characters
    than a name has (a quotation). A name in lower case, or a phrase that only reads as one, is what someone said;
    the agent can still name it with entities, and what passes and is no name stays provisional."""
    if not _valid_entity(name) or any(unicodedata.category(c) in ("So", "Sk") for c in name):
        return False  # an emoji or another picture is no part of a name guessed from text ("Alice🙂Smith")
    tokens = [t.strip(".,;:!?()'\"«»“”„") for t in name.split()]
    if any(_RE_VERSION.fullmatch(t) for t in tokens) and not any(
            any(c.isupper() or (c.isalpha() and c.lower() == c.upper()) for c in t)
            for t in tokens if not _RE_VERSION.fullmatch(t)):
        return False
    if not quoted:
        return True
    # Judged on the letters that have case, whatever else the span holds ("hrr_dim太大" too), and an identifier must
    # have a letter in it: "14 days" is a count, "gpt-4o" a name (0.16.1, issue #4)
    cased = [c for c in name if c.isalpha() and c.lower() != c.upper()]
    if cased and not any(c.isupper() for c in cased) \
            and not any(is_identifier(t) and any(c.isalpha() for c in t) for t in tokens) \
            and not (len(tokens) == 1 and _mixed(name)):  # one word glued to Chinese: 微信pay, 小米pad (review 0.16.1)
        return False
    # Every letter of a script written without spaces counts (Han, kana, Hangul, Thai...: review 0.16.1, a Japanese or
    # Thai sentence in quotes has no space and few Han), and Japanese and Korean predicates end what is said
    return sum(len(run) for run in UNSEGMENTED.findall(name)) <= _QUOTED_MAX_HAN and not is_greeting(name) \
        and not zh_phrase(name) and not ja_ko_phrase(name)


_UNRESTRICTED: set[str] = set()  # files whose owner-only mode could not be set, warned about once each


def _restrict_permissions(db_path: Path) -> None:
    """Owner-only access to the database and its WAL side files (POSIX; best effort, but a failure is logged once per
    file: the README promises 0600, and a store readable by others must not stay so without a word)."""
    if os.name != "posix":
        return
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        try:
            if path.exists() and path.stat().st_mode & 0o077:
                os.chmod(path, 0o600)
        except OSError as exc:
            if str(path) not in _UNRESTRICTED:
                _UNRESTRICTED.add(str(path))
                logger.warning("holographic: could not make %s readable by its owner only (%s)", path, exc)


class MemoryStore(HistoryMixin):
    """SQLite-backed fact store with entity resolution and trust scoring.

    Process-wide shared connection registry: SQLite allows one writer at a time and several providers
    coexist per process (main agent + every delegate_task subagent), so all instances for the same database
    share ONE connection and ONE re-entrant lock — writes are fully serialized and "database is locked" is
    impossible within a process. Refcounted: closing one instance never tears the connection out from under a
    sibling. Every multi-statement write runs in one transaction, so a failure leaves nothing half-written."""

    _shared: dict = {}
    _shared_guard = threading.Lock()

    def __init__(self, db_path: "str | Path | None" = None, default_trust: float = 0.5, hrr_dim: int = 1024, *,
                 migrate: bool = True, tz=None) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust, self.hrr_dim, self._hrr_available = _clamp_trust(default_trust), hrr_dim, hrr._HAS_NUMPY
        self.tz = tz  # local dates of the history (recorded_on, "today"); None = UTC
        try:  # resolve() so symlinked/relative paths to the same file share ONE connection
            self._key = str(self.db_path.resolve())
        except OSError:
            self._key = str(self.db_path)
        with MemoryStore._shared_guard:
            entry = MemoryStore._shared.get(self._key)
            if entry is None:
                # Autocommit outside explicit transactions (_tx): a failed statement never leaves a dangling
                # transaction holding the write lock.
                conn = sqlite3.connect(self._key, check_same_thread=False, timeout=10.0, isolation_level=None)
                conn.row_factory = sqlite3.Row
                conn.create_function("hfold", 1, fold, deterministic=True)
                conn.execute("PRAGMA secure_delete = ON")  # freed pages and cells are zeroed: forgotten text leaves no copy
                entry = MemoryStore._shared[self._key] = {"conn": conn, "lock": threading.RLock(), "refs": 0, "ready": False,
                                                          "migrated": False, "generation": 0}
            entry["refs"] += 1
            self._entry, self._conn, self._lock = entry, entry["conn"], entry["lock"]
        with self._lock:  # schema initialised once per shared connection; migrated by the first writer
            try:
                if not entry["ready"]:
                    self._init_db()
                    entry["ready"] = True
                if migrate and not entry["migrated"]:
                    self._migrate()
                    entry["migrated"] = True
                    self._maybe_rebuild_entities()
                    self.scrub()  # a forget whose file rewrite was blocked last time
                elif migrate:
                    self._check_vectors()
                    self._drop_orphan_vectors()  # facts an older version deleted while this process held the file
                    self.drop_orphan_confirmations()  # the same for confirmations (review 2026-10-15)
                    self.scrub()  # a forget whose file rewrite another process blocked: retried by every writer open
            except BaseException:
                self.close()  # drop our reference; a broken connection must not stay registered
                raise
        _restrict_permissions(self.db_path)

    def _init_db(self) -> None:
        """Create schema, enable WAL via the shared fallback helper (NFS/SMB/FUSE degrade gracefully), add hrr_vector
        to pre-HRR DBs, and narrow the FTS update trigger of DBs created by older versions."""
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="memory_store.db (holographic)")
        self._conn.executescript(_SCHEMA)
        if "hrr_vector" not in {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}:
            from hermes_cli.sqlite_util import add_column_if_missing
            add_column_if_missing(self._conn, "facts", "hrr_vector", "hrr_vector BLOB")
        trigger = self._one("SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'facts_au'")
        if trigger is not None and "UPDATE OF" not in (trigger["sql"] or "").upper():
            with self._tx():
                self._conn.execute("DROP TRIGGER facts_au")
                self._conn.execute(_FACTS_AU)

    def _migrate(self) -> None:
        """Bring the store to SCHEMA_VERSION, idempotently, on every open. Additive only, so every older version
        keeps working on the file; what an older version wrote in between is reconciled here:
        - entities it created have no ``name_key``: keys are filled in, and entities that turn out to share a key
          are merged (links and aliases moved to the oldest one);
        - aliases it may have stored in the legacy ``entities.aliases`` column are copied into ``entity_aliases``;
        - HRR vectors are re-encoded when ``hrr_dim`` differs from the dimension they were written with (recorded in
          ``meta``; before 0.4.0, inferred from the stored blobs), so a changed setting no longer breaks queries.
        0.8.0 adds the history tables (executescript commits, so before the transaction) and versions every fact; 0.14.0
        the vectors of search by meaning, whose rows for facts an older version deleted are dropped here; 0.15.0 the
        origin of a version and the confirmations of a fact, whose rows an older version's forget left are dropped;
        0.16.0 where an alias came from (entity_aliases.source: older rows and an older version's writes count as given
        by the user) and the tombstones of deleted names. The entity index is rebuilt after the migration, by
        _maybe_rebuild_entities. An older version reads such a store as before; one that writes to it as well may
        create a deleted name again from a new fact's text (its next open by this version does not notice that)."""
        self._conn.executescript(HISTORY_SCHEMA + _EMBEDDINGS_SCHEMA + _ENTITY_SCHEMA)
        with self._tx():
            if "source" not in {r[1] for r in self._conn.execute("PRAGMA table_info(entity_aliases)")}:
                # 0.16.0. 'legacy': written before, or by an older release since; nothing says whether someone gave it
                self._conn.execute("ALTER TABLE entity_aliases ADD COLUMN source TEXT DEFAULT 'legacy'")
            self._entry.pop("alias_source", None)
            self._conn.execute("DELETE FROM fact_embeddings WHERE fact_id NOT IN (SELECT fact_id FROM facts)")
            if "src_origin" not in {r[1] for r in self._conn.execute("PRAGMA table_info(fact_versions)")}:
                self._conn.execute("ALTER TABLE fact_versions ADD COLUMN src_origin TEXT")
            self._entry.pop("source_columns", None)
            self._conn.execute("DELETE FROM fact_confirmations WHERE fact_id NOT IN (SELECT fact_id FROM fact_versions)")
            if "name_key" not in {r[1] for r in self._conn.execute("PRAGMA table_info(entities)")}:
                self._conn.execute("ALTER TABLE entities ADD COLUMN name_key TEXT")
            # Aliases an older version deleted the entity of, or that point nowhere: drop before anything resolves them.
            self._conn.execute("DELETE FROM entity_aliases WHERE entity_id NOT IN (SELECT entity_id FROM entities)")
            # Legacy comma-separated aliases are imported BEFORE keys are reconciled: a merge deletes the source row.
            names = {entity_key(r["name"]) for r in self._conn.execute("SELECT name FROM entities")}
            for row in self._conn.execute("SELECT entity_id, aliases FROM entities WHERE aliases IS NOT NULL AND aliases != ''").fetchall():
                for alias in row["aliases"].split(","):
                    key = entity_key(alias)
                    if key and key not in names:
                        self._conn.execute("INSERT OR IGNORE INTO entity_aliases (alias_key, entity_id, alias) VALUES (?, ?, ?)",
                                           (key, row["entity_id"], alias.strip()))
            self._reconcile_entity_keys()
            self._rekey_entities()
            self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_key ON entities(name_key) WHERE name_key IS NOT NULL")
            self._sync_hrr_dim()
            self._migrate_history()
            self._bump()
            self._set_meta("schema_version", str(SCHEMA_VERSION))

    def _rekey_entities(self) -> None:
        """Keys and alias keys written before entity_key changed (0.16.1: ".NET" was "net", "‘Alice’" was "‘alice’") are
        moved to today's; an entity whose new key another one holds is merged into it, an alias row whose new key is
        taken goes (the other row already says it)."""
        for r in self._conn.execute("SELECT entity_id, name, name_key, entity_type FROM entities "
                                    "WHERE name_key IS NOT NULL ORDER BY entity_id").fetchall():
            key = entity_key(r["name"])
            if not key or key == r["name_key"]:
                continue
            owner = self._one("SELECT entity_id FROM entities WHERE name_key = ? AND entity_id != ?", (key, r["entity_id"]))
            if owner is None:
                self._conn.execute("UPDATE entities SET name_key = ? WHERE entity_id = ?", (key, r["entity_id"]))
            else:
                self._merge_into(r["entity_id"], owner[0], declared=r["entity_type"] == "declared")
        for r in self._conn.execute("SELECT alias_key, entity_id, alias FROM entity_aliases").fetchall():
            key = entity_key(r["alias"])
            if not key or key == r["alias_key"]:
                continue
            if self._one("SELECT 1 FROM entity_aliases WHERE alias_key = ?", (key,)) is None:
                self._conn.execute("UPDATE entity_aliases SET alias_key = ? WHERE alias_key = ?", (key, r["alias_key"]))
            else:
                self._conn.execute("DELETE FROM entity_aliases WHERE alias_key = ?", (r["alias_key"],))

    def _reconcile_entity_keys(self) -> None:
        """Fill missing name keys; merge entities whose keys collide into the oldest of them."""
        rows = self._conn.execute("SELECT entity_id, name, name_key, entity_type FROM entities ORDER BY entity_id").fetchall()
        if all(r["name_key"] for r in rows):
            return
        canonical: dict[str, int] = {}
        for r in rows:
            if r["name_key"]:
                canonical.setdefault(r["name_key"], r["entity_id"])
        for r in rows:
            if r["name_key"]:
                continue
            key = entity_key(r["name"]) or f"#{r['entity_id']}"
            keep = canonical.get(key)
            if keep is None:
                canonical[key] = r["entity_id"]
                self._conn.execute("UPDATE entities SET name_key = ? WHERE entity_id = ?", (key, r["entity_id"]))
            else:
                self._merge_into(r["entity_id"], keep, declared=r["entity_type"] == "declared")

    def _merge_into(self, source: int, target: int, *, declared: bool | None = None, alias_source: str = "user") -> None:
        """Make entity *source* part of *target* (inside a transaction): links, aliases (both representations) and
        the declared flag move, *source*'s own name stays findable as an alias (from *alias_source*: "aka" when a
        fact's "X aka Y" merged them, 0.16.0), the vectors of every affected fact are re-encoded with the surviving
        name, and category banks are dropped."""
        src = self._one("SELECT name, name_key, aliases, entity_type FROM entities WHERE entity_id = ?", (source,))
        tgt = self._one("SELECT name, name_key, aliases FROM entities WHERE entity_id = ?", (target,))
        if src is None or tgt is None or source == target:
            return
        affected = [r[0] for r in self._conn.execute("SELECT fact_id FROM fact_entities WHERE entity_id IN (?, ?)", (source, target))]
        self._conn.execute("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) SELECT fact_id, ? FROM fact_entities "
                           "WHERE entity_id = ?", (target, source))
        self._conn.execute("DELETE FROM fact_entities WHERE entity_id = ?", (source,))
        self._conn.execute("UPDATE OR IGNORE entity_aliases SET entity_id = ? WHERE entity_id = ?", (target, source))
        self._conn.execute("DELETE FROM entity_aliases WHERE entity_id = ?", (source,))
        target_key = tgt["name_key"] or entity_key(tgt["name"])
        moved = [a for a in (src["aliases"] or "").split(",") if a.strip()]
        if entity_key(src["name"]) != target_key:
            moved.append(src["name"])
            if self._alias_sources():
                self._conn.execute("INSERT OR IGNORE INTO entity_aliases (alias_key, entity_id, alias, source) "
                                   "VALUES (?, ?, ?, ?)", (entity_key(src["name"]), target, src["name"], alias_source))
            else:
                self._conn.execute("INSERT OR IGNORE INTO entity_aliases (alias_key, entity_id, alias) VALUES (?, ?, ?)",
                                   (entity_key(src["name"]), target, src["name"]))
        legacy = [a for a in (tgt["aliases"] or "").split(",") if a.strip()]
        known = {entity_key(a) for a in legacy} | {target_key}
        legacy += [a.strip() for a in moved if entity_key(a) not in known]
        self._conn.execute("UPDATE entities SET aliases = ? WHERE entity_id = ?", (_legacy_aliases(legacy), target))
        if declared if declared is not None else src["entity_type"] == "declared":
            self._conn.execute("UPDATE entities SET entity_type = 'declared' WHERE entity_id = ?", (target,))
        self._conn.execute("DELETE FROM entities WHERE entity_id = ?", (source,))
        self._reencode(affected)  # without numpy: emptied, not left with the old name's role (review 0.16.0)
        self._bump()

    def _meta(self, key: str) -> str | None:
        row = self._one("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._write("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value))

    def _bump(self) -> None:
        """Invalidate per-connection caches (the known-entity index) after an entity change or a rollback."""
        self._entry["generation"] = self._entry.get("generation", 0) + 1

    def _check_vectors(self) -> None:
        """Re-encode vectors whose size does not match hrr_dim (another version or setting wrote them)."""
        if not self._hrr_available:
            return
        sizes = (4 + 4 * self.hrr_dim, 8 * self.hrr_dim)
        stale = self._conn.execute("SELECT fact_id, content FROM facts WHERE hrr_vector IS NULL OR LENGTH(hrr_vector) NOT IN (?, ?)",
                                   sizes).fetchall()
        if stale:
            with self._tx():
                for row in stale:
                    self._compute_hrr_vector(row["fact_id"], row["content"])

    def _sync_hrr_dim(self) -> None:
        """Record hrr_dim; re-encode every vector when the configured dimension differs from the stored one."""
        if not self._hrr_available:
            return
        stored = self._meta("hrr_dim")
        if stored is None:  # before 0.4.0: infer from a stored blob (prefixed float32 or legacy float64)
            blob = self._one("SELECT hrr_vector FROM facts WHERE hrr_vector IS NOT NULL LIMIT 1")
            if blob is not None:
                data = bytes(blob["hrr_vector"])
                stored = str((len(data) - 4) // 4 if data.startswith(b"HRR1") else len(data) // 8)
        if stored is not None and stored != str(self.hrr_dim):
            logger.warning("holographic: hrr_dim changed (%s -> %s), re-encoding stored vectors", stored, self.hrr_dim)
            for row in self._conn.execute("SELECT fact_id, content FROM facts").fetchall():
                self._compute_hrr_vector(row["fact_id"], row["content"])
            self._conn.execute("DELETE FROM memory_banks")
        self._check_vectors()  # the record is not proof: an older writer may have stored other sizes since
        self._set_meta("hrr_dim", str(self.hrr_dim))

    def _one(self, sql: str, params=()):
        return self._conn.execute(sql, params).fetchone()

    def _write(self, sql: str, params=()) -> sqlite3.Cursor:
        """One write statement: autocommitted on its own, or part of the enclosing _tx()."""
        return self._conn.execute(sql, params)

    @contextlib.contextmanager
    def _tx(self):
        """One write transaction (BEGIN IMMEDIATE ... COMMIT, ROLLBACK on error); nested calls join the outer one.

        The COMMIT is inside the guard: in rollback-journal mode (the core keeps DELETE on a SQLite it deems
        WAL-reset-vulnerable, and on NFS / SMB) a reader in another process makes COMMIT fail with "database is
        locked" after the busy timeout, and a commit that failed leaves the transaction open; without the ROLLBACK
        every later _tx() would join that dangling transaction and nothing would ever be committed again."""
        with self._lock:
            if self._conn.in_transaction:
                yield
                return
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._conn.execute("COMMIT")
                if getattr(_SPLITS, "pending", None):  # committed: what collect_splits reports (0.17.1)
                    _SPLITS.found.update(_SPLITS.pending)
                    _SPLITS.pending = []
            except BaseException:
                if getattr(_SPLITS, "pending", None):
                    _SPLITS.pending = []  # rolled back: it never happened
                if self._conn.in_transaction:
                    try:
                        self._conn.execute("ROLLBACK")
                    except sqlite3.Error as exc:  # the connection is broken: the error raised below says so
                        logger.warning("holographic: rollback after a failed write failed too (%s)", exc)
                self._bump()  # anything cached inside the transaction never happened
                raise

    def read(self, sql: str, params=()) -> list:
        """Run a read under the store lock (a consistent snapshot against this process's writers)."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def add_fact(self, content: str, category: str = "general", tags: str = "") -> int:
        """Insert a fact and return its fact_id; on duplicate content (UNIQUE) return the existing fact_id untouched."""
        return self.add_fact_status(content, category=category, tags=tags)[0]

    def add_fact_status(self, content: str, category: str = "general", tags: str = "", *,
                        trust: float | None = None, entities=(), valid_from=None,
                        source: dict | None = None) -> tuple[int, bool]:
        """(fact_id, created). Links extracted, declared (*entities*) and already known entities and stores the HRR
        vector and the first version (*valid_from*: the date it started to hold, None = unknown; *source*: where it
        was written) in the same transaction. *trust* overrides the default trust of a new fact."""
        content = (content or "").strip()
        if not content:
            raise ValueError("content must not be empty")
        start = parse_day(valid_from, field="valid_from", today=self._today())
        with self._tx():
            try:
                fact_id: int = self._write(
                    "INSERT INTO facts (content, category, tags, trust_score) VALUES (?, ?, ?, ?)",
                    (content, category, tags, _clamp_trust(self.default_trust if trust is None else trust))).lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                existing = int(self._one("SELECT fact_id FROM facts WHERE content = ?", (content,))["fact_id"])
                self._declare(existing, entities)  # who it is about still counts (0.17.0; dropped until 0.16.3)
                return existing, False
            self._link_entities(fact_id, content, entities)
            self._compute_hrr_vector(fact_id, content)
            self._invalidate_bank(category)
            if self.history_ready():
                self._insert_version(fact_id, content, category, tags, valid_from=start, source=source)
            return fact_id, True

    def update_fact(self, fact_id: int, content: str | None = None, trust_delta: float | None = None,
                    tags: str | None = None, category: str | None = None, *, entities=(), touch: bool = True,
                    source: dict | None = None) -> bool:
        """Partially update a fact (trust clamped to [0, 1]). Returns True if the row existed. Raises ValueError for
        empty content and DuplicateContentError when another fact already has the new content. touch=False keeps
        updated_at (temporal decay and "recently changed" read it): for search aids that do not change the fact.

        A new text is a correction (as history.correct_fact): the old text stays as a `corrected` version. Tags and
        category change the current version in place."""
        if content is not None:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")
        with self._tx():
            row = self._one("SELECT fact_id, trust_score, category, content, tags FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                return False
            history = self.history_ready()
            if history:
                self._reconcile_history()
            if content is not None and history and content != row["content"]:
                other = self._one("SELECT fact_id FROM facts WHERE content = ? AND fact_id != ?", (content, fact_id))
                if other is not None:
                    raise DuplicateContentError(int(other["fact_id"]))
                cur = self._current_version(fact_id)
                self._replace_current(fact_id, "corrected", content, category or row["category"],
                                      row["tags"] if tags is None else tags, valid_from=cur["valid_from"],
                                      source=self._with_origin(source, cur), trust=row["trust_score"])
            elif history and (tags is not None or category is not None):
                self._write("UPDATE fact_versions SET tags = COALESCE(?, tags), category = COALESCE(?, category) "
                            "WHERE fact_id = ? AND status = 'current'", (tags, category, fact_id))
            changes = {col: val for col, val in {
                "content": content, "tags": tags, "category": category,
                "trust_score": _clamp_trust(row["trust_score"] + trust_delta) if trust_delta is not None else None,
            }.items() if val is not None}
            assignments = ", ".join((["updated_at = CURRENT_TIMESTAMP"] if touch else []) + [f"{col} = ?" for col in changes])
            if not assignments and not entities:
                return True
            try:
                if assignments:  # entities alone (touch=False) change no column (review 0.17.0: they were dropped)
                    self._write(f"UPDATE facts SET {assignments} WHERE fact_id = ?", [*changes.values(), fact_id])
            except sqlite3.IntegrityError:
                raise self._duplicate(content) from None
            if content is not None:  # re-extract entities (declared links survive a rewording) and recompute the vector
                self._relink_fact(fact_id, content, entities)
                self._compute_hrr_vector(fact_id, content)
                self._drop_orphan_entities()
            elif entities:  # declare who the fact is about without rewording it
                current = self._one("SELECT content FROM facts WHERE fact_id = ?", (fact_id,))["content"]
                self._link_declared(fact_id, entities)
                self._compute_hrr_vector(fact_id, current)
            self._invalidate_bank(row["category"])
            if category is not None and category != row["category"]:
                self._invalidate_bank(category)
            return True

    def remove_fact(self, fact_id: int) -> bool:
        """0.7.0 `remove`, now `forget` (history.forget_fact): the fact, its history, links and entities left without
        facts are deleted for good and a tombstone is kept. Returns True if the fact existed (current or ended)."""
        if not self.history_ready():  # a store no writer of 0.8.0 has migrated: the plain delete
            with self._tx():
                row = self._one("SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,))
                if row is None:
                    return False
                self._delete_fact_row(fact_id, row["category"])
                return True
        try:
            return not self.forget_fact(fact_id).get("already")
        except KeyError:
            return False

    # -- rows of the current projection (inside a transaction) ---------------------------------------------------

    def _delete_fact_row(self, fact_id: int, category: str | None) -> None:
        if self._entry.get("migrated"):  # the vector goes with the fact (forget, end, remove)
            self._write("DELETE FROM fact_embeddings WHERE fact_id = ?", (fact_id,))
        self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
        self._write("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
        self._drop_orphan_entities()
        self._invalidate_bank(category)

    def _rewrite_fact(self, fact_id: int, content: str, category, tags, *, entities=(), old_category=None,
                      reset_trust: bool = False, trust: float | None = None) -> None:
        """The current text of a fact changed (change / correct / revert): the row, its entity links (declared ones
        survive), vector and banks follow. reset_trust: a changed fact starts over (default trust, no counts)."""
        sets = ["content = ?", "category = ?", "tags = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list = [content, category or "general", tags or ""]
        if reset_trust:  # *trust*, when given, is where it starts over (its origin's, 0.15.0)
            sets += ["trust_score = ?", "helpful_count = 0", "retrieval_count = 0"]
            params.append(self.default_trust if trust is None else _clamp_trust(trust))
        elif trust is not None:
            sets.append("trust_score = ?")
            params.append(_clamp_trust(trust))
        try:
            self._write(f"UPDATE facts SET {', '.join(sets)} WHERE fact_id = ?", [*params, fact_id])
        except sqlite3.IntegrityError:
            raise self._duplicate(content) from None
        self._relink_fact(fact_id, content, entities)
        self._compute_hrr_vector(fact_id, content)
        self._drop_orphan_entities()
        self._invalidate_bank(old_category)
        if category != old_category:
            self._invalidate_bank(category)

    def _duplicate(self, content: str) -> DuplicateContentError:
        """The error for a write that would give a second fact *content* (the UNIQUE column refused it)."""
        other = self._one("SELECT fact_id FROM facts WHERE content = ?", (content,))
        return DuplicateContentError(int(other["fact_id"]) if other else -1)

    def _relink_fact(self, fact_id: int, content: str, entities=()) -> None:
        """Link a fact to its entities again from *content*: names someone declared on it stay, *entities* are
        declared with it (a new text, or the same text under other names)."""
        kept = [r[0] for r in self._conn.execute(
            "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
            "WHERE fe.fact_id = ? AND e.entity_type = 'declared'", (fact_id,))]
        self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
        self._link_entities(fact_id, content, [*kept, *(entities or ())])

    def _insert_fact_row(self, fact_id: int, version, *, trust: float, entities=()) -> None:
        """Put an ended fact back into the current projection under its own id (resume / revert of an end)."""
        first = self._one("SELECT MIN(recorded_at) FROM fact_versions WHERE fact_id = ?", (fact_id,))[0]
        try:
            self._write("INSERT INTO facts (fact_id, content, category, tags, trust_score, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)", (fact_id, version["content"], version["category"], version["tags"],
                                                      _clamp_trust(trust), first or self._now()))
        except sqlite3.IntegrityError:
            raise self._duplicate(version["content"]) from None
        self._link_entities(fact_id, version["content"], entities)
        self._compute_hrr_vector(fact_id, version["content"])
        self._invalidate_bank(version["category"])

    # -- vectors of search by meaning (embeddings.py) -------------------------------------------------------------

    def put_vectors(self, model: str, rows) -> int:
        """Store vectors made by *model* from (fact_id, text hash, blob); a fact that is gone, or whose text changed
        while its vector was being made, is skipped. Returns how many were stored."""
        stored = 0
        with self._tx():
            for fact_id, digest, blob in rows:
                row = self._one("SELECT content FROM facts WHERE fact_id = ?", (fact_id,))
                if row is None or text_hash(row["content"]) != digest:
                    continue
                self._write("INSERT INTO fact_embeddings (fact_id, model, text_hash, vector, revision) "
                            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(fact_id) DO UPDATE SET model = excluded.model, "
                            "text_hash = excluded.text_hash, vector = excluded.vector, revision = excluded.revision, "
                            "created_at = CURRENT_TIMESTAMP", (fact_id, model, digest, blob, os.urandom(8).hex()))
                stored += 1
        return stored

    def _drop_orphan_vectors(self) -> None:
        """Vectors of facts that are gone (an older version deleted them: it knows no vector table)."""
        with self._tx():
            self._write("DELETE FROM fact_embeddings WHERE fact_id NOT IN (SELECT fact_id FROM facts)")

    def drop_vectors(self) -> int:
        """Delete every vector (search by meaning is off: nothing derived from the facts by an outside model stays in
        the file). Returns how many there were."""
        with self._lock:
            if not self._entry.get("migrated") or not self._one("SELECT 1 FROM fact_embeddings LIMIT 1"):
                return 0
            with self._tx():
                return max(self._write("DELETE FROM fact_embeddings").rowcount, 0)

    def list_facts(self, category: str | None = None, min_trust: float = 0.0, limit: int = 50) -> list[dict]:
        """Browse facts ordered by trust_score descending, optionally filtered by category / min trust."""
        category_clause = "AND category = ? " if category is not None else ""
        params = [min_trust] + ([category] if category is not None else []) + [limit]
        sql = ("SELECT fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, "
               f"created_at, updated_at FROM facts WHERE trust_score >= ? {category_clause}"
               "ORDER BY trust_score DESC LIMIT ?")
        return [dict(r) for r in self.read(sql, params)]

    def count_facts(self) -> int:
        return int(self.read("SELECT COUNT(*) FROM facts")[0][0])

    def record_retrievals(self, fact_ids) -> int:
        """retrieval_count += 1 for facts handed to the model; returns the number of rows touched.

        Only the counter changes: ``updated_at`` drives temporal decay and recall is not an edit (and the FTS
        trigger ignores the column). Only integer ids count (``True`` and ``1.9`` do not); duplicates count once."""
        ids = sorted({int(raw) for raw in fact_ids or ()
                      if (isinstance(raw, int) and not isinstance(raw, bool)) or (isinstance(raw, str) and raw.isdigit())})
        ids = ids[:_MAX_COUNTED_IDS]
        if not ids:
            return 0
        with self._lock:
            cur = self._write(f"UPDATE facts SET retrieval_count = retrieval_count + 1 "
                              f"WHERE fact_id IN ({','.join('?' * len(ids))})", ids)
            return max(cur.rowcount, 0)

    def add_retrieval_counts(self, counts: dict) -> bool:
        """Add buffered recall counts ({fact_id: n}) in one transaction without waiting on another process's write
        lock (busy timeout 0 for this write only). False when the database was busy: the caller keeps the counts."""
        rows = [(int(n), int(i)) for i, n in counts.items()
                if isinstance(i, int) and not isinstance(i, bool) and isinstance(n, int) and n > 0]
        if not rows:
            return True
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout = 0")
            try:
                with self._tx():
                    self._conn.executemany("UPDATE facts SET retrieval_count = retrieval_count + ? WHERE fact_id = ?", rows)
                return True
            except sqlite3.OperationalError as exc:
                logger.debug("holographic: recall counts deferred (%s)", exc)
                return False
            finally:
                self._conn.execute("PRAGMA busy_timeout = 10000")

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Adjust trust asymmetrically: helpful -> +0.05 and helpful_count += 1; unhelpful -> -0.10.
        Returns {fact_id, old_trust, new_trust, helpful_count}. Raises KeyError if fact_id is unknown."""
        with self._tx():
            row = self._one("SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                raise NotFound(f"fact_id {fact_id} not found")
            old_trust: float = row["trust_score"]
            new_trust = _clamp_trust(old_trust + (_HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA))
            increment = 1 if helpful else 0
            self._write("UPDATE facts SET trust_score = ?, helpful_count = helpful_count + ?, "
                        "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?", (new_trust, increment, fact_id))
            return {"fact_id": fact_id, "old_trust": old_trust, "new_trust": new_trust, "helpful_count": row["helpful_count"] + increment}

    def entity_ids(self, name: str) -> list[int]:
        """Entities whose name or an alias has the identity key of *name* (see entity_key). Entities an older
        version created since this store was opened have no key yet; they are matched by casefolded name."""
        key = entity_key(name or "")
        if not key:
            return []
        with self._lock:
            if not self._keyed():  # a file no writer of 0.4.0 or later has migrated: names only
                return sorted({r[0] for r in self._conn.execute("SELECT entity_id FROM entities WHERE hfold(name) = ?", (fold(name),))})
            ids = {r[0] for r in self._conn.execute(
                "SELECT entity_id FROM entities WHERE name_key = ? "
                "UNION SELECT ea.entity_id FROM entity_aliases ea JOIN entities e ON e.entity_id = ea.entity_id WHERE ea.alias_key = ? "
                "UNION SELECT entity_id FROM entities WHERE name_key IS NULL AND hfold(name) = ?", (key, key, fold(name)))}
        return sorted(ids)

    def add_alias(self, name: str, alias: str, *, source: str = "user") -> dict:
        """Make *alias* another name of the entity *name* (created if needed). If *alias* is already an entity of its
        own, it is merged into *name*: links and aliases move, and its name becomes an alias; *name* stays the name
        (rename_entity changes it). *source* (0.16.0): "user" for the alias action, "aka" for "X aka Y" found in a
        fact's text. Only an alias someone gave makes the entity established, keeps it without facts and brings back
        a name delete_entity removed."""
        if not isinstance(name, str) or not isinstance(alias, str):
            raise ValueError("an alias needs two names")
        if source == "user":
            name, alias = _clean_given(name), _clean_given(alias)
        key, alias_key = entity_key(name), entity_key(alias)
        if not key or not alias_key or key == alias_key:
            raise ValueError("an alias needs two different non-empty names")
        for side in (name, alias):  # a name someone gives passes what every given name does (review 0.16.0)
            if not _valid_entity(side.strip(), given=True):
                raise ValueError(f"'{side.strip()[:80]}' cannot be a name (2-{_ENTITY_MAX_CHARS} characters, mostly "
                                 "letters; no paths, URLs, key=value, commas or flags)")
        with self._tx():
            if source == "user":
                textual = self._text_alias(name)  # alias {entity: <a name only a text gave>} (0.17.0)
                if textual is not None:
                    raise ValueError(self._text_alias_refusal(name, textual))
                self._lift_tombstones(name, alias)
            target = self._resolve_entity(name.strip())
            own = self._one("SELECT name FROM entities WHERE entity_id = ?", (target,))
            if alias_key == entity_key(own["name"]):  # alias("Bob", "Alice") with Bob already Alice's: no-op (review)
                return {"entity_id": target, "alias": alias.strip(), "merged": []}
            owner = self._one("SELECT ea.entity_id, e.name FROM entity_aliases ea JOIN entities e ON e.entity_id = ea.entity_id "
                              "WHERE ea.alias_key = ?", (alias_key,))
            was = self._alias_row(alias) if source == "user" else None
            if owner is not None and owner["entity_id"] != target:
                if was is not None and was["source"] == "aka":
                    raise ValueError(f"'{alias.strip()}' is a name only a fact's text gave '{owner['name']}' (\"X aka "
                                     f"Y\"), and nobody confirmed it. If it belongs to '{name.strip()}' instead: split "
                                     f"{{entity: '{alias.strip()}'}} first, then alias")
                raise ValueError(f"'{alias.strip()}' is already an alias of '{owner['name']}'; merge the two entities "
                                 f"(alias '{name.strip()}' -> '{owner['name']}') if they are the same")
            confirmed = was is not None and was["source"] == "aka" and was["entity_id"] == target
            other = [r[0] for r in self._conn.execute("SELECT entity_id FROM entities WHERE name_key = ? AND entity_id != ?",
                                                      (alias_key, target))]
            for merged in other:
                self._merge_into(merged, target, alias_source=source)
            self._put_alias(alias_key, target, alias.strip(), source)
            # Older versions resolve aliases from the comma-separated column: keep it in step.
            row = self._one("SELECT aliases FROM entities WHERE entity_id = ?", (target,))
            legacy = [a for a in (row["aliases"] or "").split(",") if a.strip()]
            if alias_key not in {entity_key(a) for a in legacy}:
                self._conn.execute("UPDATE entities SET aliases = ? WHERE entity_id = ?", (_legacy_aliases(legacy + [alias.strip()]), target))
            self._bump()
            result = {"entity_id": target, "alias": alias.strip(), "merged": other}
            if confirmed:  # a name a fact's text gave, now confirmed by someone (0.17.0)
                result["confirmed"] = True
            return result

    def _put_alias(self, alias_key: str, entity_id: int, alias: str, source: str = "user") -> None:
        """Insert or move an alias row. An alias someone gave stays theirs when the same name is found in text later."""
        if self._entry.get("migrated"):  # the source column is added by this version's migration
            self._conn.execute("INSERT INTO entity_aliases (alias_key, entity_id, alias, source) VALUES (?, ?, ?, ?) "
                               "ON CONFLICT(alias_key) DO UPDATE SET entity_id = excluded.entity_id, alias = excluded.alias, "
                               "source = CASE WHEN entity_aliases.source = 'user' THEN 'user' ELSE excluded.source END",
                               (alias_key, entity_id, alias, source))
        else:
            self._conn.execute("INSERT INTO entity_aliases (alias_key, entity_id, alias) VALUES (?, ?, ?) "
                               "ON CONFLICT(alias_key) DO UPDATE SET entity_id = excluded.entity_id, alias = excluded.alias",
                               (alias_key, entity_id, alias))

    def _keyed(self) -> bool:
        """Whether the file has the entity keys of 0.4.0 (name_key, and entity_aliases kept in step): a writer of 0.4.0
        or later migrated it at some point. Reads use them in every context, also one that does not migrate — a
        subagent found no entity by an alias before (review 0.16.0); writes still wait for this process's migration."""
        if self._entry.get("migrated"):
            return True
        known = self._entry.get("keyed")
        if known is None:
            with self._lock:
                known = self._entry["keyed"] = "name_key" in {
                    r[1] for r in self._conn.execute("PRAGMA table_info(entities)")}
        return known

    def _alias_sources(self) -> bool:
        """Whether entity_aliases says where each alias came from (a store this version migrated)."""
        known = self._entry.get("alias_source")
        if known is None:
            with self._lock:
                known = self._entry["alias_source"] = "source" in {
                    r[1] for r in self._conn.execute("PRAGMA table_info(entity_aliases)")}
        return known

    def _alias_row(self, name: str):
        """The entity_aliases row of *name*'s key, with where it came from ('legacy' when the file does not say) and
        the name of the entity it belongs to; None when no alias has that key or the file has no keys (0.17.0)."""
        key = entity_key(name or "")
        if not key or not self._keyed():
            return None
        source = "COALESCE(ea.source, 'legacy')" if self._alias_sources() else "'legacy'"
        with self._lock:
            return self._one(f"SELECT ea.alias_key, ea.entity_id, ea.alias, {source} AS source, e.name AS owner "
                             "FROM entity_aliases ea JOIN entities e ON e.entity_id = ea.entity_id WHERE ea.alias_key = ?",
                             (key,))

    def _text_alias(self, name: str):
        """The alias row when *name* names one entity only through an alias that "X aka Y" in a fact's text gave and
        nobody confirmed, else None (0.17.0). Such a name is a hint for text, never the target of an action someone
        takes with a name: a declaration splits it off (_link_declared), alias, rename and delete_entity refuse it."""
        row = self._alias_row(name)
        if row is None or row["source"] != "aka" or self.entity_ids(name) != [row["entity_id"]]:
            return None
        return row

    @staticmethod
    def _text_alias_refusal(name: str, row) -> str:
        return (f"'{name.strip()}' is a name only a fact's text gave '{row['owner']}' (\"X aka Y\"), and nobody "
                f"confirmed it. If it is the same one: alias {{entity: '{row['owner']}', content: '{row['alias']}'}}; "
                f"if they are two: split {{entity: '{row['alias']}'}}; then try again")

    @contextlib.contextmanager
    def collect_splits(self):
        """{declared name: what split_alias answers} for every alias a fact's text gave that a name declared during
        this operation split off (0.17.0). Read it only after the operation succeeded; nested in one thread, the
        outer one collects."""
        outer = getattr(_SPLITS, "found", None)
        if outer is not None:
            yield outer
            return
        found: dict = {}
        _SPLITS.found, _SPLITS.pending = found, []
        try:
            yield found
        finally:
            _SPLITS.found, _SPLITS.pending = None, []

    def _drop_alias_row(self, alias_key: str, entity_id: int) -> None:
        """Remove one alias from both representations (the entity_aliases row and the legacy column)."""
        self._write("DELETE FROM entity_aliases WHERE alias_key = ? AND entity_id = ?", (alias_key, entity_id))
        row = self._one("SELECT aliases FROM entities WHERE entity_id = ?", (entity_id,))
        if row is not None and row["aliases"]:
            kept = [a for a in row["aliases"].split(",") if a.strip() and entity_key(a) != alias_key]
            self._write("UPDATE entities SET aliases = ? WHERE entity_id = ?", (_legacy_aliases(kept), entity_id))

    def _detach_alias(self, row, *, declared: bool = True, declaring: int | None = None) -> dict:
        """Make the alias *row* (see _alias_row) an entity of its own, inside the caller's transaction (0.17.0).

        1. The alias leaves the entity it belonged to (both representations) and becomes an entity (declared).
        2. Aliases "X aka Y" gave that no current fact backs any more are dropped, again and again until none is
           left: in a chain "Alice aka Bob", "Bob aka Carl", "Carl aka Dan", Carl hung on Bob and Dan on Carl; one
           pass left Dan with Alice (review 0.17.0, Codex P1.2). Only aliases are dropped here, no link yet.
        3. The facts of every entity that lost an alias are linked again from their text, the facts that say
           "X aka Y" first (so a chain is given again to whoever its text now names: "Bob aka Carl" makes Carl Bob's
           before "Carl likes tea" is read). A link to such an entity stays when the text names it; when it does not,
           a declared one stays unless the text names one of the names it lost (that name is now someone else's:
           "Bob aka Carl" no longer keeps Alice, review 0.17.0 Codex P1.1), and a fact it stays on whose text names it
           not is listed under "facts_to_check": why a declared link exists the file does not record. Not the fact
           *declaring* when it is being declared on the owner too ("They met", entities=[Alice, Bob]).
        An owner someone gave the alias to (alias {entity: X, content: Y}) was named by that person: it counts as
        declared from here, so it does not go when Y takes its facts (review 0.17.0, Fable: X vanished with its alias).
        The answer is read from the links at the end (review 0.17.0: it described a state in between); "from_dropped"
        says the old entity went with its last fact."""
        owner, alias, alias_key = row["entity_id"], row["alias"], row["alias_key"]
        lost: dict[int, set[str]] = {owner: {alias}}  # entity -> names it lost
        self._drop_alias_row(alias_key, owner)
        if row["source"] == "user":
            self._write("UPDATE entities SET entity_type = 'declared' WHERE entity_id = ?", (owner,))
        new = self._resolve_entity(alias)
        if declared:
            self._write("UPDATE entities SET entity_type = 'declared' WHERE entity_id = ?", (new,))
        self._bump()
        for _ in range(self._one("SELECT COUNT(*) FROM entity_aliases")[0] + 1):
            rows = self._conn.execute("SELECT alias_key, entity_id, alias FROM entity_aliases WHERE source = 'aka'").fetchall()
            backed = self._aka_backing(r["entity_id"] for r in rows) if rows else set()
            gone = [r for r in rows if (r["entity_id"], r["alias_key"]) not in backed]
            if not gone:
                break
            for r in gone:
                self._drop_alias_row(r["alias_key"], r["entity_id"])
                lost.setdefault(r["entity_id"], set()).add(r["alias"])
            self._bump()
        kinds = {r[0]: r[1] for r in self._conn.execute(
            f"SELECT entity_id, entity_type FROM entities WHERE entity_id IN ({','.join('?' * len(lost))})",
            tuple(lost)).fetchall()}
        affected = sorted({r[0] for r in self._conn.execute(
            f"SELECT fact_id FROM fact_entities WHERE entity_id IN ({','.join('?' * len(lost))})", tuple(lost)).fetchall()})
        texts = {r[0]: r[1] for r in self._conn.execute(
            f"SELECT fact_id, content FROM facts WHERE fact_id IN ({','.join('?' * len(affected))})",
            tuple(affected)).fetchall()} if affected else {}
        before = {f: {r[0] for r in self._conn.execute("SELECT entity_id FROM fact_entities WHERE fact_id = ?", (f,))}
                  for f in affected}
        check: set[int] = set()
        for fact_id in sorted(affected, key=lambda f: (not self._aka_pairs(texts.get(f) or ""), f)):
            content = texts.get(fact_id)
            if content is None:
                continue
            current = {r[0]: r[1] for r in self._conn.execute(
                "SELECT fe.entity_id, e.entity_type FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
                "WHERE fe.fact_id = ?", (fact_id,)).fetchall()}
            named = self._text_entity_ids(content) | self.mentioned_entity_ids(content)
            links = set(named) | {e for e, kind in current.items() if kind == "declared" and e not in lost}
            for entity_id in lost:
                if entity_id not in current or entity_id in named or kinds.get(entity_id) != "declared":
                    continue
                moved_to = {i for name in lost[entity_id] for i in self.entity_ids(name)} - {entity_id}
                if not named & moved_to:
                    links.add(entity_id)  # declared there, and the text says nothing about the names it lost
                    check.add(fact_id)
            self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
            for entity_id in sorted(links):
                self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (fact_id, entity_id))
        self._reencode(affected)
        self._drop_orphan_entities(keep_declared=True)
        self._bump()
        after = {f: {r[0] for r in self._conn.execute("SELECT entity_id FROM fact_entities WHERE fact_id = ?", (f,))}
                 for f in affected}
        out = {"entity": alias, "entity_id": new, "from": row["owner"], "was": row["source"],
               "facts_moved": [f for f in affected if owner in before[f] and owner not in after[f]],
               "facts_shared": [f for f in affected if owner in after[f] and new in after[f]],
               "facts_to_check": sorted(f for f in check if owner in after.get(f, ()) and f != declaring)}
        if self._one("SELECT 1 FROM entities WHERE entity_id = ?", (owner,)) is None:
            out["from_dropped"] = True
        return out

    def split_alias(self, name: str) -> dict:
        """Make an alias an entity of its own (the split action, 0.17.0): one a fact's text gave, one someone gave or
        one of unknown origin; the new entity counts as declared. See _detach_alias for the facts. KeyError for an
        unknown name; ValueError for an entity's own name or a name of two entities."""
        self._require_writer()
        if not isinstance(name, str) or not name.strip():
            raise ValueError("split needs the alias in 'entity'")
        with self._tx():
            ids = self.entity_ids(name)
            if not ids:
                raise NotFound(f"no entity is named '{name.strip()}'")
            if len(ids) > 1:
                raise ValueError(f"'{name.strip()}' names {len(ids)} entities; fact_store entities shows them")
            row = self._alias_row(name)
            if row is None or row["entity_id"] != ids[0]:
                raise ValueError(f"'{name.strip()}' is the name of an entity, not an alias; rename or delete_entity "
                                 "changes a name")
            self._lift_tombstones(name)  # someone named it, as a declaration does (review 0.17.0)
            return self._detach_alias(row)

    def unconfirmed_text_aliases(self, limit: int = 20) -> list[dict]:
        """Aliases "X aka Y" in a fact's text gave that nobody confirmed (0.17.0), with the facts that say them: for
        review. All of them, however old: what matters is that a person looks, not when the text said it."""
        if not self._alias_sources():
            return []
        out = []
        for r in self.read("SELECT ea.alias, ea.alias_key, ea.entity_id, e.name AS owner FROM entity_aliases ea "
                           "JOIN entities e ON e.entity_id = ea.entity_id WHERE ea.source = 'aka' "
                           "ORDER BY e.name, ea.alias")[:limit]:
            facts = []
            for f in self.read("SELECT f.fact_id, f.content FROM facts f JOIN fact_entities fe ON fe.fact_id = f.fact_id "
                               "WHERE fe.entity_id = ? AND (f.content LIKE '%aka%' OR f.content LIKE '%a.k.a.%' "
                               "OR f.content LIKE '%also%known%as%') ORDER BY f.fact_id", (r["entity_id"],)):
                if any(entity_key(right) == r["alias_key"] and r["entity_id"] in self.entity_ids(left)
                       for left, right in self._aka_pairs(f["content"])):
                    facts.append(f["fact_id"])
            out.append({"entity": r["owner"], "alias": r["alias"], "facts": facts,
                        "reason": "a name a fact's text gave (\"X aka Y\"); nobody confirmed it is the same one"})
        return out

    def _entity_rows(self) -> list[dict]:
        """Every entity with its aliases, number of facts and standing, most linked first (0.16.0). Established: someone
        named it (declared with entities=[...], renamed, or given an alias with the alias action), or two or more
        current facts are linked to it; provisional: a name the extraction rules guessed from one fact, nobody
        confirmed and no other fact named. A standing, not a verdict: it drops back when the second fact goes.
        A name without capitals (Chinese, Japanese, Thai...) that the facts only ever put in quotes stays provisional
        however many facts quote it (0.16.2, issue #4): Chinese quotes a button, a menu item or a step as readily as a
        name, and two facts quoting "自动添加任务并解析元数据" made it established. Then it takes a person naming it, or
        two linked facts of which one names it outside plain quotes — in title marks 《》, in "X aka Y", or in
        running text as mentions are found (_used_outside_quotes)."""
        given = _GIVEN_ALIAS if self._alias_sources() else "1"
        rows = self.read("SELECT e.entity_id, e.name, e.entity_type, e.created_at, COUNT(fe.fact_id) AS facts FROM entities e "
                         "LEFT JOIN fact_entities fe ON fe.entity_id = e.entity_id GROUP BY e.entity_id "
                         "ORDER BY facts DESC, e.name")
        aliases: dict[int, list[str]] = {}
        textual: dict[int, list[str]] = {}  # aliases "X aka Y" in a fact's text gave, nobody confirmed (0.17.0)
        named: set[int] = set()
        for r in self.read(f"SELECT entity_id, alias, {given} AS given FROM entity_aliases ORDER BY alias"):
            aliases.setdefault(r["entity_id"], []).append(r["alias"])
            if r["given"]:
                named.add(r["entity_id"])
            elif self._alias_sources():
                textual.setdefault(r["entity_id"], []).append(r["alias"])
        outside = self._used_outside_quotes({
            r["entity_id"]: aliases.get(r["entity_id"], []) for r in rows
            if r["facts"] >= 2 and r["entity_id"] not in named and r["entity_type"] != "declared" and _uncased(r["name"])})
        out = []
        for r in rows:
            declared = r["entity_type"] == "declared"
            established = declared or r["entity_id"] in named or (
                r["facts"] >= 2 and (not _uncased(r["name"]) or r["entity_id"] in outside))
            out.append({"entity_id": r["entity_id"], "name": r["name"], "facts": r["facts"], "declared": declared,
                        "aliases": aliases.get(r["entity_id"], []),
                        "text_aliases": textual.get(r["entity_id"], []),
                        "status": "established" if established else "provisional",
                        "_since": r["created_at"] or ""})
        return out

    def _used_outside_quotes(self, candidates: dict) -> set[int]:
        """Of *candidates* ({entity_id: its aliases}), those a linked current fact names outside plain quotes (see
        _entity_rows): on the fact's text with plain quotes blanked (_outside_plain_quotes), what the linking readers
        find — mentioned_entity_ids, and the names _extract_entities finds (title marks, "X aka Y") resolved by
        entity_ids — or an alias with case written there as a word, compared as entity keys compare (Straße =
        STRASSE: a one-word alias is not looked up in running text, review 0.16.2). Each fact's text is read once
        and kept, with what it names, for the current facts only and until an entity changes (review 0.16.2: 1,000
        names on two long facts took 11 s; a forgotten fact's text must not stay in memory)."""
        if not candidates:
            return set()
        with self._lock:
            version = (self._entry.get("generation", 0), self._one("PRAGMA data_version")[0])
            cached = self._entry.get("outside")
            previous = cached[1] if cached is not None and cached[0] == version else {}
            current: dict[int, tuple] = {}
            links: dict[int, list[int]] = {}
            ordered = sorted(candidates)
            for k in range(0, len(ordered), _MAX_COUNTED_IDS):
                chunk = ordered[k:k + _MAX_COUNTED_IDS]
                for row in self._conn.execute(
                        f"SELECT entity_id, fact_id FROM fact_entities WHERE entity_id IN ({','.join('?' * len(chunk))})",
                        chunk):
                    links.setdefault(row["fact_id"], []).append(row["entity_id"])
            fact_ids = sorted(links)
            for k in range(0, len(fact_ids), _MAX_COUNTED_IDS):
                chunk = fact_ids[k:k + _MAX_COUNTED_IDS]
                for row in self._conn.execute(
                        f"SELECT fact_id, content FROM facts WHERE fact_id IN ({','.join('?' * len(chunk))})", chunk):
                    content = row["content"] or ""
                    seen = previous.get(row["fact_id"])
                    if seen is None or seen[0] != content:
                        bare = _outside_plain_quotes(content)
                        ids = set(self.mentioned_entity_ids(bare))
                        for name in self._extract_entities(bare):
                            ids.update(self.entity_ids(name))
                        seen = (content, bare.casefold(), frozenset(ids))
                    current[row["fact_id"]] = seen
            self._entry["outside"] = (version, current)
            found: set[int] = set()
            for fact_id, (_, bare, ids) in current.items():
                for entity_id in links[fact_id]:
                    if entity_id in ids or any(not _uncased(a) and _cased_word_in(
                            bare, unicodedata.normalize("NFKC", a).casefold()) for a in candidates[entity_id]):
                        found.add(entity_id)
            return found

    def list_entities(self, limit: int = 50, status: str | None = None) -> list[dict]:
        """Entities with their aliases, number of facts and standing (see _entity_rows), most linked first. *status*
        "established" or "provisional" keeps those only; provisional ones come oldest first, the first to clean up."""
        if status not in (None, "all", "established", "provisional"):
            raise ValueError("status must be 'all', 'established' or 'provisional'")
        rows = self._entity_rows()
        if status in ("established", "provisional"):
            rows = [r for r in rows if r["status"] == status]
        if status == "provisional":
            rows.sort(key=lambda r: (r["_since"], r["entity_id"]))
        return [{k: v for k, v in r.items() if k != "_since"} for r in rows[:limit]]

    def entity_counts(self) -> dict:
        """{"entities": all, "provisional": guessed names nobody confirmed (see _entity_rows), "text_aliases": aliases
        a fact's text gave that nobody confirmed (0.17.0)}."""
        rows = self._entity_rows()
        return {"entities": len(rows), "provisional": sum(r["status"] == "provisional" for r in rows),
                "text_aliases": sum(len(r["text_aliases"]) for r in rows)}

    def stale_provisional_entities(self, days: int = 60, limit: int = 20) -> list[dict]:
        """Provisional entities older than *days*, oldest first: clean-up candidates for review (0.16.0)."""
        old = {r[0] for r in self.read("SELECT entity_id FROM entities WHERE julianday('now') - julianday(created_at) > ?",
                                       (days,))}
        rows = sorted((r for r in self._entity_rows() if r["status"] == "provisional" and r["entity_id"] in old),
                      key=lambda r: (r["_since"], r["entity_id"]))
        return [{"entity_id": r["entity_id"], "entity": r["name"], "facts": r["facts"],
                 "reason": (f"a name guessed from one fact; nobody declared it or named it again in {days}+ days"
                            if r["facts"] < 2 else
                            f"found only in quotes, in {r['facts']} facts; nobody declared it or used it as a name in "
                            f"{days}+ days")}
                for r in rows[:limit]]

    def _extract_entities(self, text: str) -> list[str]:
        """Entity candidates (see the pattern table) shaped like a name (_name_shaped) and not deleted (delete_entity),
        deduplicated by casefold in first-seen order. The text is read in NFC, so a letter written with a combining
        accent is one letter ("José" either way), and with code spans and blocks blanked out (0.16.1)."""
        text = _without_code(unicodedata.normalize("NFC", text or ""))
        quoted = _straight_quoted(text) + [m.group(1) for pattern in _RE_QUOTED for m in pattern.finditer(text)]
        found = [(n, False) for n in _capitalized_phrases(text)] + [
            (q, True) for q in quoted if len(q) <= _QUOTED_MAX_CHARS and len(q.split()) <= _QUOTED_MAX_WORDS]
        for left, right in self._aka_pairs(text):
            found += [(left, False), (right, False)]
        uniq: dict[str, str] = {}  # casefolded key -> first-seen spelling, insertion-ordered
        for raw, quoted in found:
            name = raw.strip(" ,;:!?").rstrip(".") if raw else ""  # a leading dot stays: ".NET" (0.16.1)
            if name and fold(name) not in uniq and _name_shaped(name, quoted=quoted) and not self._tombstoned(name):
                uniq[fold(name)] = name
        return list(uniq.values())

    def _aka_pairs(self, text: str) -> list[tuple[str, str]]:
        """(name, alias) for the first _AKA_MAX "X aka Y" in *text* (NFC, code blanked: see _extract_entities) whose
        sides are shaped like names and not deleted.
        Each side is read within _AKA_WINDOW characters of its "aka": reading the whole text before and after every
        "aka" made a 10,000-character chain of them take a second per fact (review 0.16.0). A name is at most
        _ENTITY_MAX_CHARS long, far under the window, so a run the window cuts is too long to be a name anyway."""
        text = _without_code(unicodedata.normalize("NFC", text or ""))
        pairs = []
        for m in itertools.islice(_RE_AKA.finditer(text), _AKA_MAX):
            before, after = text[max(0, m.start() - _AKA_WINDOW):m.start()], text[m.end():m.end() + _AKA_WINDOW]
            left_text, right_text = before.rstrip(), after.lstrip()
            if not (_inline_gap(before[len(left_text):]) and _inline_gap(after[:len(after) - len(right_text)])):
                continue  # "aka" on a line of its own
            left = _aka_side(left_text, from_end=True)
            right = _aka_side(right_text, from_end=False)
            left, right = (n.strip(" ,;:!?").rstrip(".") if n else "" for n in (left, right))
            if text_fold(left.lower()) in STOPWORDS or text_fold(right.lower()) in STOPWORDS:
                continue  # "Bruce Banner aka the Hulk": "the" is no alias (review 0.16.1)
            if _name_shaped(left, quoted=_mixed(left)) and _name_shaped(right, quoted=_mixed(right)) \
                    and entity_key(left) != entity_key(right) and not self._tombstoned(left) and not self._tombstoned(right):
                pairs.append((left, right))
        return pairs

    def _text_entity_ids(self, content: str, declared_keys=frozenset()) -> set[int]:
        """Entities *content* itself names: extracted names (created if new) and "X aka Y" (Y an alias of X rather than
        a second entity, unless Y is already one, or is a name declared with this very fact: *declared_keys*, 0.17.0)."""
        pairs = self._aka_pairs(content)
        alias_only = {entity_key(right) for _, right in pairs}
        ids = {self._resolve_entity(name) for name in self._extract_entities(content)
               if entity_key(name) not in alias_only or self.entity_ids(name)}
        for left, right in pairs:
            if not self.entity_ids(right) and entity_key(right) not in declared_keys:
                ids.add(self.add_alias(left, right, source="aka")["entity_id"])
        return ids

    def _link_entities(self, fact_id: int, content: str, declared=()) -> None:
        """Link the fact to its entities: names extracted from the text, names the caller declared (a one-word name
        such as "Masha" is accepted here), and entities already known that the text mentions in any word form."""
        declared_keys = frozenset(entity_key(_clean_given(n)) for n in declared or ())
        ids = self._text_entity_ids(content, declared_keys) | self.mentioned_entity_ids(content)
        for entity_id in sorted(ids):
            self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (fact_id, entity_id))
        self._link_declared(fact_id, declared)

    def _declare(self, fact_id: int, declared) -> None:
        """Declare names on a fact whose text stays as it is (0.17.0: an add of a text already stored, a correct that
        changes no text, a change into a text another fact holds): until 0.16.3 such declarations were dropped."""
        if not declared:
            return
        self._link_declared(fact_id, declared)
        self._reencode([fact_id])  # its roles changed: the vector again (emptied without numpy), banks go (0.17.1)

    def _link_declared(self, fact_id: int, declared) -> None:
        """Link names the caller declared (one-word names allowed) and mark them declared. A declared name is someone
        naming it: a tombstone of delete_entity on it is lifted. A name that is only an alias "X aka Y" in a fact's
        text gave (nobody confirmed it) is not resolved to X: it becomes an entity of its own first (0.17.0, see
        _detach_alias; the operation's collect_splits says so), since a text must not decide who a name someone gives
        belongs to."""
        for name in declared or ():
            name = _clean_given(name)
            if not _valid_entity(name, given=True):
                logger.info("holographic: a declared name of %d characters is no name; not linked", len(name))
            else:
                self._lift_tombstones(name)
                row = self._text_alias(name) if self._entry.get("migrated") else None
                if row is not None:
                    also = any(row["entity_id"] in self.entity_ids(_clean_given(other))
                               for other in declared if _clean_given(other) != name)
                    detached = self._detach_alias(row, declaring=fact_id if also else None)
                    if getattr(_SPLITS, "found", None) is not None:  # reported once the transaction commits
                        _SPLITS.pending.append((name, detached))
                entity_id = self._resolve_entity(name)
                if self._write("UPDATE entities SET entity_type = 'declared' WHERE entity_id = ? AND entity_type != 'declared'",
                               (entity_id,)).rowcount:
                    self._bump()
                self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (fact_id, entity_id))

    def _known_entities(self) -> dict[str, list[tuple[int, tuple]]]:
        """{key of the first word: [(entity_id, key sets of a name's words)]} for entities safe to match in free text:
        multi-word names, and one-word names only when declared (older stores hold one-word junk such as a quoted
        "Done", which would otherwise attach itself to every fact saying "done"). Cached per connection and rebuilt
        after any entity change here, a rollback, or a commit by another connection (PRAGMA data_version)."""
        with self._lock:
            version = (self._entry.get("generation", 0), self._one("PRAGMA data_version")[0])
            cached = self._entry.get("entities")
            if cached is None or cached[0] != version:
                index: dict[str, list[tuple[int, tuple]]] = {}
                names: dict[int, list[str]] = {}
                if self._keyed():
                    for row in self._conn.execute("SELECT ea.entity_id, ea.alias FROM entity_aliases ea "
                                                  "JOIN entities e ON e.entity_id = ea.entity_id"):
                        names.setdefault(row["entity_id"], []).append(row["alias"])
                for row in self._conn.execute("SELECT entity_id, name, aliases, entity_type FROM entities"):
                    for name in [row["name"], *((row["aliases"] or "").split(",")), *names.get(row["entity_id"], [])]:
                        name = unicodedata.normalize("NFKC", name)  # as mentioned_entity_ids reads text
                        if name.strip().startswith(".") and len(name.strip()) > 1:  # ".NET": with its dot, below
                            index.setdefault(".", []).append((int(row["entity_id"]), name.strip().lower()))
                            continue
                        key = name_keys(name)
                        if len(key) > 1 or (key and row["entity_type"] == "declared"):
                            for first in key[0][0]:
                                index.setdefault(first, []).append((int(row["entity_id"]), key))
                        glued = name.strip().lower()
                        if UNSEGMENTED.search(glued) and len(glued) >= 2 and (len(key) > 1 or row["entity_type"] == "declared"):
                            index.setdefault("", []).append((int(row["entity_id"]), glued))  # "" = substring names
                cached = self._entry["entities"] = (version, index)
            return cached[1]

    def mentioned_entity_ids(self, text: str) -> set[int]:
        """Known entities whose name (or alias) occurs in *text* as a run of words: a Latin-script name as written
        (or with 's), a name in another script by its keys, so "Маши" and "Машу" mention "Маша" and "لإيفا" mentions
        "إيفا", while "jeans" does not mention "Jean" and "машина" not "Маша". Looked up by each word's keys, not by
        scanning names. The text is read in NFKC, as names are indexed, and stretch by stretch between separators
        (0.16.1): the words of one name stand together on one line, never across a line break, a comma or a bracket
        ("Alice\nSmith" and "Alice, Smith" do not mention Alice Smith)."""
        text = unicodedata.normalize("NFKC", text or "")
        index = self._known_entities()
        found = set()
        for stretch in _mention_stretches(text):
            sequence = analyze(stretch).words
            for i, word in enumerate(sequence):
                for first in word.keys:
                    for entity_id, key in index.get(first, ()):
                        if entity_id not in found and len(sequence) - i >= len(key) and all(
                                sequence[i + j].text in forms if exact else not forms.isdisjoint(sequence[i + j].keys)
                                for j, (forms, exact) in enumerate(key)):
                            found.add(entity_id)
        lowered_text = text.lower()
        for entity_id, name in index.get(".", ()):  # ".NET" with its dot, not the word "net" (0.16.1)
            if entity_id not in found and re.search(r"(?<![\w.])" + re.escape(name) + r"(?!\w)", lowered_text):
                found.add(entity_id)
        substring_names = index.get("", ())
        if substring_names and UNSEGMENTED.search(text):  # a name in Chinese/Thai text has no spaces around it
            if is_chinese(text):
                # From word cut to word cut only (0.16.1: 和服 was found in 我和服务员, "I and the waiter"), Traditional
                # and Simplified as one
                folded = analyze(text).text
                cuts = zh_word_cuts(folded)
                for entity_id, name in substring_names:
                    if entity_id not in found and _found_between_cuts(folded, text_fold(name), cuts):
                        found.add(entity_id)
            else:
                lowered = text.lower()
                for entity_id, name in substring_names:
                    if name in lowered:
                        found.add(entity_id)
        return found

    def _resolve_entity(self, name: str) -> int:
        """Return the entity_id for a casefold name or alias match, creating the entity if absent."""
        ids = self.entity_ids(name)
        if ids:
            return ids[0]
        self._bump()
        if not self._entry.get("migrated"):
            return int(self._write("INSERT INTO entities (name) VALUES (?)", (name,)).lastrowid)  # type: ignore[arg-type]
        return int(self._write("INSERT INTO entities (name, name_key) VALUES (?, ?)", (name, entity_key(name))).lastrowid)  # type: ignore[arg-type]

    def _drop_orphan_entities(self, keep_aliased: bool = True, keep_declared: bool = False) -> None:
        """Entities no fact links to go — but not one the user gave an alias (alias may come before any fact names
        it; an unrelated rewrite used to wipe it with its aliases). Forget drops the names its fact gave itself.
        First, aliases "X aka Y" gave (0.16.0) go with the last current fact that says so (forget, end, a new text): an
        alias is the text's, and a forgotten fact must not live on as one; such an alias keeps nothing. *keep_declared*:
        a declared entity stays without facts too (the rebuild leaves what someone named as it is)."""
        self._prune_aka_aliases()
        keep = (f" AND entity_id NOT IN (SELECT entity_id FROM entity_aliases WHERE {_GIVEN_ALIAS})"
                if keep_aliased and self._entry.get("migrated") else "")
        if keep_declared:
            keep += " AND COALESCE(entity_type, '') != 'declared'"
        if self._write("DELETE FROM entities WHERE entity_id NOT IN (SELECT entity_id FROM fact_entities)"
                       + keep).rowcount:
            self._bump()
        if self._entry.get("migrated"):
            self._write("DELETE FROM entity_aliases WHERE entity_id NOT IN (SELECT entity_id FROM entities)")

    def _aka_backing(self, entity_ids) -> set[tuple[int, str]]:
        """(entity_id, alias key) of every "X aka Y" that a current fact linked to one of *entity_ids* says."""
        out: set[tuple[int, str]] = set()
        for entity_id in sorted(set(entity_ids)):
            for row in self._conn.execute(
                    "SELECT f.content FROM facts f JOIN fact_entities fe ON fe.fact_id = f.fact_id WHERE fe.entity_id = ? "
                    "AND (f.content LIKE '%aka%' OR f.content LIKE '%a.k.a.%' OR f.content LIKE '%also%known%as%')",
                    (entity_id,)).fetchall():
                for left, right in self._aka_pairs(row["content"]):
                    if entity_id in self.entity_ids(left):
                        out.add((entity_id, entity_key(right)))
        return out

    def _prune_aka_aliases(self, legacy_keys=frozenset()) -> None:
        """Drop aliases "X aka Y" gave that no current fact says any more, from both representations; with
        *legacy_keys* (the "Y" a forgotten fact said), also aliases of unknown origin ('legacy') with those keys. The
        facts the alias linked are linked again from their text, so a name only the gone text gave (the "X") does not
        stay on them either: it then goes as an entity without facts (review 0.16.0)."""
        if not self._entry.get("migrated"):
            return
        rows = self._conn.execute("SELECT alias_key, entity_id, alias, source FROM entity_aliases WHERE source = 'aka'"
                                  + (f" OR (source = 'legacy' AND alias_key IN ({','.join('?' * len(legacy_keys))}))"
                                     if legacy_keys else ""), tuple(sorted(legacy_keys))).fetchall()
        if not rows:
            return
        backed = self._aka_backing(r["entity_id"] for r in rows)
        gone = [r for r in rows if (r["entity_id"], r["alias_key"]) not in backed]
        for r in gone:
            self._drop_alias_row(r["alias_key"], r["entity_id"])
        if gone:
            self._bump()
            self._relink({f for r in gone for f, in self._conn.execute(
                "SELECT fact_id FROM fact_entities WHERE entity_id = ?", (r["entity_id"],)).fetchall()})

    def _aka_alias_keys(self, texts) -> set[str]:
        """Keys of the "Y" of every "X aka Y" in *texts* (what a forgotten fact said; see _prune_aka_aliases)."""
        return {entity_key(right) for text in texts if text for _, right in self._aka_pairs(text)}

    def _relink(self, fact_ids) -> None:
        """Link facts to their entities again from their own text, as a new text does (declared links stay)."""
        for fact_id in sorted(set(fact_ids)):
            row = self._one("SELECT content FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                continue
            self._relink_fact(fact_id, row["content"])
        self._reencode(fact_ids)

    # -- names someone removed, renamed or wants indexed again (0.16.0) ------------------------------------------------

    def _require_writer(self) -> None:
        """delete_entity, rename_entity and rebuild_entity_index write what only a migrated store holds (tombstones,
        where an alias came from, the scrub mark): a store opened without migrating refuses them (review 0.16.0: it
        answered "scrub": "done" for a file it had not rewritten)."""
        if not self._entry.get("migrated"):
            raise ValueError("this store was opened read-only (without migrating); entity changes need the main agent")

    def _name_digest(self, name: str) -> str:
        """A deleted name is kept as a keyed hash, as the text of a forgotten fact is: the file never says it."""
        return self.digest("entity " + entity_key(name))

    def _name_digests(self, name: str) -> set[str]:
        """The digests a deleted name is kept under: its key, and its key folded as search folds words (Traditional and
        Simplified Chinese as one, accents dropped), so 台湾大学 is suppressed after 臺灣大學 was deleted (review 0.16.1).
        The first is what 0.16.0 wrote."""
        key = entity_key(name)
        return {self.digest("entity " + k) for k in (key, text_fold(key), _entity_key_0160(name))}

    def _tombstoned(self, name: str) -> bool:
        """Whether delete_entity removed this name: text never creates it again."""
        if not self._entry.get("migrated") or not entity_key(name) \
                or self._one("SELECT 1 FROM entity_tombstones LIMIT 1") is None:
            return False
        digests = sorted(self._name_digests(name))
        return self._one(f"SELECT 1 FROM entity_tombstones WHERE digest IN ({','.join('?' * len(digests))})",
                         digests) is not None

    def _lift_tombstones(self, *names: str) -> None:
        """Someone named it (entities=[...], alias, rename): a deleted name may be created from text again."""
        if not self._entry.get("migrated") or self._one("SELECT 1 FROM entity_tombstones LIMIT 1") is None:
            return
        for name in names:
            if entity_key(name or ""):
                for digest in sorted(self._name_digests(name)):
                    self._write("DELETE FROM entity_tombstones WHERE digest = ?", (digest,))

    def _reencode(self, fact_ids) -> int:
        """Vectors of facts whose entities changed: computed again; without numpy emptied, so that the next writer that
        has it computes them (_check_vectors) instead of keeping roles of names that are gone. Banks go too."""
        done = 0
        for fact_id in sorted(set(fact_ids)):
            row = self._one("SELECT content FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                continue
            if self._hrr_available:
                self._compute_hrr_vector(fact_id, row["content"])
            else:
                self._write("UPDATE facts SET hrr_vector = NULL WHERE fact_id = ?", (fact_id,))
            done += 1
        self._write("DELETE FROM memory_banks")
        return done

    def delete_entity(self, name: str) -> dict:
        """Delete an entity by its name or an alias: the name, its aliases and its links to facts. The facts stay as
        they are: this cleans the index of names and forgets nothing (fact_store forget does that). The name and its
        aliases get a tombstone, so a fact's text never creates them again; an explicit name (entities=[...], alias,
        rename) lifts it. Aliases "X aka Y" gave go without one ("text_aliases_dropped", 0.17.1): a text decided they
        were X's, and Y may be someone else. The file is rewritten afterwards, as after forget, so freed pages keep no copy of a name no
        fact says ("scrub": "pending" when another process blocked it; the next open retries). KeyError when no entity
        has that name or alias."""
        result = self._delete_entity(name)
        result["scrub"] = "done" if self.scrub() else "pending"
        return result

    def delete_entities(self, names) -> dict:
        """delete_entity for several names in one transaction and one rewrite of the file (0.16.2, issue #4: cleaning
        up 23 provisional names took 23 calls). *names*: a list of 1 to _DELETE_MAX non-empty strings, checked whole
        before anything changes (review 0.16.2: a list with a number in it, or a dict, deleted the names it held).
        Each name must name one entity: one that names two (a name of one and an alias of another, in a store an older
        version wrote) refuses the call, as an unknown name alone does not. Answer: "deleted" (as delete_entity),
        "covered" (names of an entity this call deleted already: an alias given after its name), "not_found"; KeyError
        when none is known, ValueError for input that is no such list or for a name of two entities."""
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError("entities must be a list of names")
        if len(names) > _DELETE_MAX:
            raise ValueError(f"at most {_DELETE_MAX} names at once, {len(names)} were given")
        if not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError("every name in entities must be a non-empty string")
        wanted = list(dict.fromkeys(n.strip() for n in names))
        self._require_writer()
        deleted, covered, missing = [], [], []
        with self._tx():
            plan, ambiguous, textual = [], [], []
            for name in wanted:
                ids = self.entity_ids(name)
                if not ids:
                    missing.append(name)
                elif len(ids) > 1:
                    ambiguous.append(name)
                else:
                    if self._text_alias(name) is not None:
                        textual.append(name)
                    plan.append((name, ids[0]))
            if ambiguous:
                raise ValueError("these names belong to more than one entity: " + ", ".join(f"'{n}'" for n in ambiguous)
                                 + "; fact_store entities shows them — delete each by a name or alias only it has")
            if textual:  # checked whole, before anything changes (0.17.0)
                raise ValueError(self._text_alias_refusal(textual[0], self._text_alias(textual[0]))
                                 + (f" (also: {', '.join(repr(n) for n in textual[1:])})" if textual[1:] else ""))
            if not plan:
                raise NotFound("no entity is named " + ", ".join(f"'{n}'" for n in missing))
            done: set[int] = set()
            for name, entity_id in plan:
                if entity_id in done:
                    covered.append(name)
                    continue
                deleted.append(self._delete_entity(name))
                done.add(entity_id)
        return {"deleted": deleted, "covered": covered, "not_found": missing,
                "scrub": "done" if self.scrub() else "pending"}

    def _delete_entity(self, name: str) -> dict:
        self._require_writer()
        with self._tx():
            ids = self.entity_ids(name or "")
            if not ids:
                raise NotFound(f"no entity is named '{(name or '').strip()}'")
            if len(ids) > 1:  # one entity's name and another's alias, in a store an older version wrote (review 0.16.2)
                raise ValueError(f"'{name.strip()}' belongs to more than one entity; fact_store entities shows them — "
                                 "delete each by a name or alias only it has")
            textual = self._text_alias(name)  # a text must not decide whose name is deleted (0.17.0)
            if textual is not None:
                raise ValueError(self._text_alias_refusal(name, textual))
            names: list[str] = []
            textual_names: list[str] = []  # aliases "X aka Y" gave: they go with the entity, without a tombstone
            facts: set[int] = set()
            aliases = 0
            source = "COALESCE(source, 'legacy')" if self._alias_sources() else "'legacy'"
            for entity_id in ids:
                row = self._one("SELECT name, aliases FROM entities WHERE entity_id = ?", (entity_id,))
                rows = self._conn.execute(f"SELECT alias, {source} FROM entity_aliases WHERE entity_id = ?",
                                          (entity_id,)).fetchall()
                own = [r[0] for r in rows]
                legacy = [a.strip() for a in (row["aliases"] or "").split(",") if a.strip()]
                # A text decided that "Y" is X's, not someone: deleting X must not keep Y from ever being found again
                # (review 0.17.0, Fable: "Bob Brown lives in Berlin" then linked no entity at all).
                text_keys = {entity_key(r[0]) for r in rows if r[1] == "aka"} - {entity_key(row["name"])}
                textual_names += [a for a in dict.fromkeys(a for a in (*own, *legacy) if entity_key(a) in text_keys)]
                names += [n for n in (row["name"], *own, *legacy) if entity_key(n) not in text_keys]
                aliases += len({entity_key(a) for a in (*own, *legacy)} - {entity_key(row["name"])})
                facts |= {r[0] for r in self._conn.execute("SELECT fact_id FROM fact_entities WHERE entity_id = ?", (entity_id,))}
                self._write("DELETE FROM fact_entities WHERE entity_id = ?", (entity_id,))
                self._write("DELETE FROM entity_aliases WHERE entity_id = ?", (entity_id,))
                self._write("DELETE FROM entities WHERE entity_id = ?", (entity_id,))
            if self._entry.get("migrated"):
                for digest in sorted({d for n in names if entity_key(n) for d in self._name_digests(n)}):
                    self._write("INSERT OR IGNORE INTO entity_tombstones (digest) VALUES (?)", (digest,))
                self._set_meta("scrub_pending", "1")
            self._reencode(facts)
            self._bump()
            result = {"entity": names[0], "deleted": True, "facts_unlinked": len(facts), "aliases_dropped": aliases,
                      "facts_deleted": 0}
            if textual_names:
                result["text_aliases_dropped"] = sorted({entity_key(n): n for n in textual_names}.values())
            return result

    def rename_entity(self, name: str, new_name: str) -> dict:
        """Give an entity another name; its id, links and aliases stay, the old name becomes an alias, and the entity
        counts as declared (someone named it). Only another spelling ("phoenix" -> "Phoenix") changes the name alone;
        the entity's own alias swaps places with its name. A name another entity has, as its name or an alias, is
        refused: two entities become one only through alias, never through a rename (a typo would merge two people).
        KeyError for an unknown name; ValueError for an empty or unusable new name, or one that names two entities."""
        self._require_writer()
        new = _clean_given(new_name)
        new_key = entity_key(new)
        if not new_key or not _valid_entity(new, given=True):
            raise ValueError(f"'{str(new_name)[:80]}' cannot be a name (2-{_ENTITY_MAX_CHARS} characters, mostly letters; "
                             "no paths, URLs, key=value, commas or flags)")
        with self._tx():
            ids = self.entity_ids(name or "")
            if not ids:
                raise NotFound(f"no entity is named '{(name or '').strip()}'")
            if len(ids) > 1:
                raise ValueError(f"'{name.strip()}' names {len(ids)} entities; rename one by a name only it has")
            textual = self._text_alias(name)  # a text must not decide whose name is changed (0.17.0)
            if textual is not None:
                raise ValueError(self._text_alias_refusal(name, textual))
            entity_id = ids[0]
            taken = self._text_alias(new)  # the hint below would merge two people a text tied (review 0.17.0, Fable)
            if taken is not None and taken["entity_id"] != entity_id:
                raise ValueError(self._text_alias_refusal(new, taken))
            others = [i for i in self.entity_ids(new) if i != entity_id]
            if others:
                other = self._one("SELECT name FROM entities WHERE entity_id = ?", (others[0],))["name"]
                raise ValueError(f"'{new}' already names the entity '{other}'. If both are one, alias {{entity: '{other}', "
                                 f"content: '{name.strip()}'}} makes them one and keeps '{other}' as the name; rename "
                                 "after that if the other name should stay.")
            row = self._one("SELECT name, aliases FROM entities WHERE entity_id = ?", (entity_id,))
            old = row["name"]
            old_key = entity_key(old)
            legacy = [a.strip() for a in (row["aliases"] or "").split(",") if a.strip()]
            self._lift_tombstones(new)
            if new_key != old_key:
                self._write("DELETE FROM entity_aliases WHERE alias_key = ? AND entity_id = ?", (new_key, entity_id))
                self._put_alias(old_key, entity_id, old)
                legacy = [a for a in legacy if entity_key(a) != new_key]
                if old_key not in {entity_key(a) for a in legacy}:
                    legacy.append(old)
            self._write("UPDATE entities SET name = ?, aliases = ?, entity_type = 'declared' WHERE entity_id = ?",
                        (new, _legacy_aliases(legacy), entity_id))
            if self._entry.get("migrated"):
                self._write("UPDATE entities SET name_key = ? WHERE entity_id = ?", (new_key, entity_id))
            self._reencode([r[0] for r in self._conn.execute("SELECT fact_id FROM fact_entities WHERE entity_id = ?",
                                                             (entity_id,))])
            self._bump()
            aliases = [r[0] for r in self._conn.execute("SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY alias",
                                                        (entity_id,))]
            return {"entity_id": entity_id, "name": new, "previous": old, "aliases": aliases}

    def entity_rules(self) -> int:
        """The extraction rules the store's entity index was built with (1: before 0.16.0, which recorded none)."""
        try:
            return int(self._meta("entity_rules") or 1)
        except ValueError:
            return 1

    def rebuild_entity_index(self) -> dict:
        """Link every fact to its entities again under this release's extraction rules: names an older release
        guessed that these rules do not, and links only they gave, go. What someone named stays as it is: declared
        entities, entities with an alias someone gave (or of unknown origin), and their links. First pass: the facts
        in the order they were stored, each read as adding it would read it — so "X aka Y" makes Y an alias of X only
        while no earlier fact has named Y and nobody gave Y; a text never merges a name already in use into another
        (review 0.16.0: a stored "Mallory Jones aka Alice Smith" must not take over the facts about Alice Smith).
        Second pass: the known names each fact mentions, against the whole index the first left, so an earlier fact
        gets a name a later one brought. Entities that stay keep their id and age; vectors are made again only for
        facts whose entities changed. Records the rules. ValueError when a newer release indexed the store."""
        self._require_writer()
        with self._tx():
            if self.entity_rules() > _ENTITY_RULES:
                raise ValueError(f"a newer release indexed this store's entities (rules {self.entity_rules()}, this one "
                                 f"knows {_ENTITY_RULES}); it is left as it is")
            before = self._one("SELECT COUNT(*) FROM entities")[0]
            given = f" WHERE {_GIVEN_ALIAS}" if self._alias_sources() else ""
            protected = {r[0] for r in self._conn.execute("SELECT entity_id FROM entities WHERE entity_type = 'declared'")}
            protected |= {r[0] for r in self._conn.execute(f"SELECT entity_id FROM entity_aliases{given}")}
            old = {(r[0], r[1]) for r in self._conn.execute("SELECT fact_id, entity_id FROM fact_entities")}
            # A name two facts held before the rebuild is no "Y" an "X aka Y" may take: fact ids stay through change
            # and update, so "earlier id" is not "said earlier" (review 0.16.1). Not kept by this alone: junk two
            # facts held still goes when the rules no longer guess it.
            held = Counter(entity_id for _, entity_id in old)
            established = {entity_id for entity_id, n in held.items() if n >= 2}
            facts = self._conn.execute("SELECT fact_id, content FROM facts ORDER BY fact_id").fetchall()
            first: set[tuple[int, int]] = set()
            named: set[int] = set()  # entities an earlier fact of this pass named
            for f in facts:
                for left, right in self._aka_pairs(f["content"]):
                    # Y left over from the old index and not named yet: the alias adding this fact would have made
                    if not set(self.entity_ids(right)) & (protected | named | established):
                        try:
                            self.add_alias(left, right, source="aka")
                        except ValueError:  # Y is another entity's alias already
                            pass
                ids = self._text_entity_ids(f["content"])
                named |= ids
                first |= {(f["fact_id"], e) for e in ids}
            current = {(r[0], r[1]) for r in self._conn.execute("SELECT fact_id, entity_id FROM fact_entities")}
            kept = {link for link in current if link[1] in protected}
            for link in sorted(current - first - kept):  # merges above moved links: diff against the table, not *old*
                self._write("DELETE FROM fact_entities WHERE fact_id = ? AND entity_id = ?", link)
            for link in sorted(first - current):
                self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", link)
            self._drop_orphan_entities(keep_declared=True)  # entities only a dropped link held
            self._bump()
            second = {(f["fact_id"], e) for f in facts for e in self.mentioned_entity_ids(f["content"])}
            for link in sorted(second - first - kept):
                self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", link)
            new = {(r[0], r[1]) for r in self._conn.execute("SELECT fact_id, entity_id FROM fact_entities")}
            vectors = self._reencode({fact_id for fact_id, _ in old ^ new})
            self._set_meta("entity_rules", str(_ENTITY_RULES))
            self._bump()
            return {"facts": len(facts), "entities_before": before,
                    "entities_after": self._one("SELECT COUNT(*) FROM entities")[0],
                    "links_removed": len(old - new), "links_added": len(new - old), "vectors": vectors}

    def entity_index_state(self) -> str:
        """"current", or what a store indexed under older rules (too large to rebuild on open) needs."""
        rules = self.entity_rules()
        if rules >= _ENTITY_RULES:
            return "current"
        return f"built with extraction rules {rules} of {_ENTITY_RULES}: run fact_store rebuild_entities"

    def _maybe_rebuild_entities(self) -> None:
        """On a writer's first open: a store indexed under older rules is rebuilt, up to _AUTO_REBUILD_MAX_FACTS facts
        (the write lock is held throughout, and other processes wait at most 10 s for it); a larger one says so in
        stats and is rebuilt on request. A failure leaves the old index and is logged: the store works as before."""
        if self.entity_rules() >= _ENTITY_RULES:
            return
        facts, chars = self._one("SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0) FROM facts")
        if facts > _AUTO_REBUILD_MAX_FACTS or chars > _AUTO_REBUILD_MAX_CHARS:
            logger.info("holographic: %d facts, %d characters; the entity index is rebuilt on request "
                        "(fact_store rebuild_entities)", facts, chars)
            return
        try:
            result = self.rebuild_entity_index()
        except Exception as exc:  # noqa: BLE001 - the old index works; never fail an open over it
            logger.warning("holographic: entity index not rebuilt (%s); the old one stays", exc)
            return
        if result["links_removed"] or result["links_added"]:
            logger.info("holographic: entity index rebuilt under rules %d: %d -> %d entities, %d links removed, %d added",
                        _ENTITY_RULES, result["entities_before"], result["entities_after"], result["links_removed"],
                        result["links_added"])

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store the HRR vector for a fact (linked entities as roles). No-op without numpy."""
        if not self._hrr_available:
            return
        entities = [row["name"] for row in self._conn.execute(_ENTITY_NAMES_SQL, (fact_id,)).fetchall()]
        blob = hrr.phases_to_bytes(hrr.encode_fact(content, entities, self.hrr_dim))
        self._write("UPDATE facts SET hrr_vector = ? WHERE fact_id = ?", (blob, fact_id))

    def _invalidate_bank(self, category: str | None) -> None:
        """Drop the category's memory bank. Retrieval no longer reads banks; deleting (instead of rebuilding the
        whole category on every write) keeps a downgrade safe: an older version that finds no bank falls back to
        the per-fact vectors, and rebuilds the bank itself on its next write."""
        if category is not None:
            self._write("DELETE FROM memory_banks WHERE bank_name = ?", (f"cat:{category}",))

    @classmethod
    def release_all_under(cls, directory: "str | Path") -> int:
        """Force-close every shared connection whose database lives under ``directory``; returns the count.
        close() is refcount-driven, so a live holder (e.g. an agent's provider) keeps a profile's SQLite handle
        open, which on Windows makes rmtree of the profile fail. The directory is going away, so later use by a
        stale holder is expected to fail.

        That is exactly what a profile delete must break on Windows: the desktop's main ``serve`` process
        opens ``memory_store.db`` for every known profile, and ``rmtree`` of the profile directory fails
        with ``WinError 32`` while any of those handles is open (#88347). In a process that holds none (e.g.
        the CLI deleting from outside serve) this is a harmless no-op returning 0.
        """
        root = os.path.normcase(str(Path(directory).expanduser().resolve())) + os.sep
        with cls._shared_guard:
            doomed = [cls._shared.pop(key) for key in list(cls._shared) if os.path.normcase(key).startswith(root)]
            for entry in doomed:
                try:
                    with entry["lock"]:
                        entry["conn"].close()
                except Exception:
                    pass  # an already-closed/broken connection must not abort releasing siblings
        return len(doomed)

    def close(self) -> None:
        """Release this instance's reference; the connection closes with the last holder. Idempotent."""
        with MemoryStore._shared_guard:
            entry = getattr(self, "_entry", None)
            if entry is None:
                return
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                try:
                    entry["conn"].close()
                finally:
                    # Pop only OUR entry: after release_all_under() a same-path store may have
                    # registered a FRESH entry under this key; a stale late close() must not evict it.
                    # See #88347.
                    if MemoryStore._shared.get(self._key) is entry:
                        MemoryStore._shared.pop(self._key, None)
            self._entry = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
