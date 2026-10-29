"""Two questions in one message (prefetch_clause_coverage: an experiment, off by default).

The coverage gate weighs a fact against every informative word of the message. A message that asks two things at once
(«что Злате нужно обязательно взять с собой на плавание в четверг и какой чай она любит?») gives each answer only its
half of the words: the fact about tea holds «Злате», «чай» and «любит», every word of its question, and still covers
0.24 of the message, under the gate's 0.3, so it is left out. With prefetch_clause_coverage on, a fact the gate left
out gets a second look against each of the message's two questions alone. It is let in only when it holds at least two
words of that question itself and covers the question's words (a word no fact holds counts against it, as in the gate),
and when the question is about a person — named in it, or «она» / «он» of a name in the other question — only if the
fact names that person: «Тимур любит зелёный чай» answers «какой чай она любит» only when «она» is Тимур. Whose answer
it is must be told for sure (Codex review 2026-10-29): «она» of one of two people, a name the store does not know, a word
that is also a known name («чай» when a dog is called «Чай») — nothing is let in. A rescued fact comes after the facts
the gate kept and, with the reranker on too, only takes a place the others leave free, so it never takes the place of
one of them. A message is split only outside quotes and only into two questions that each stand alone. The facts are
synthetic; the household is invented."""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider, rerank

# key -> (text, declared names). Several people like tea, so that the second question has answers about the wrong
# person; the words of the questions are spread over the other facts, so that IDF is that of a household store.
FACTS = {
    "zlata_swim": ("По четвергам у Златы плавание; в четверг напомнить взять очки и шапочку.", ["Злата"]),
    "zlata_tea": ("Злата любит чай с бергамотом.", ["Злата"]),
    "zlata_dance": ("Злата ходит на танцы по понедельникам.", ["Злата"]),
    "zlata_allergy": ("У Златы аллергия на орехи.", ["Злата"]),
    "zlata_bd": ("День рождения Златы — 14 февраля.", ["Злата"]),
    "timur_tea": ("Тимур любит зелёный чай с мятой.", ["Тимур"]),
    "timur_football": ("Футбол у Тимура по субботам в десять утра; в субботу взять бутсы и форму.", ["Тимур"]),
    "timur_juice": ("Тимур любит яблочный сок.", ["Тимур"]),
    "timur_bd": ("День рождения Тимура — 3 мая.", ["Тимур"]),
    "timur_school": ("Тимур учится в седьмом классе.", ["Тимур"]),
    "vera_tea": ("Вера пьёт чай без сахара по утрам.", ["Вера"]),
    "vera_books": ("Вера любит читать детективы перед сном.", ["Вера"]),
    "vera_work": ("Вера работает в банке до шести вечера.", ["Вера"]),
    "gleb_coffee": ("Глеб пьёт кофе с молоком.", ["Глеб"]),
    "gleb_work": ("Глеб работает из дома по пятницам.", ["Глеб"]),
    "gleb_tea": ("Глеб не пьёт чай вечером.", ["Глеб"]),
    "ira_tea": ("Ира любит травяной чай.", ["Ира"]),
    "ira_yoga": ("У Иры йога по средам; в среду взять коврик.", ["Ира"]),
    "nina_pills": ("Бабушке Нине нужно пить таблетки от давления утром.", ["Нина"]),
    "musya_food": ("Кошке Мусе покупать влажный корм.", ["Муся"]),
    "tea_shop": ("Чай покупаем в лавке на углу.", []),
    "trash": ("Мусор вывозят во вторник и в пятницу.", []),
    "school": ("Школа у детей начинается в восемь утра.", []),
    "umbrella": ("В сезон дождей брать с собой зонт.", []),
    "market": ("Рынок работает по выходным до двух часов.", []),
    "dentist": ("Стоматолог принимает по средам после обеда.", []),
    "plants": ("Орхидеи поливать раз в неделю.", []),
    "car": ("Машину заправлять на станции у рынка.", []),
    "pool_bag": ("Сумку для бассейна хранить в прихожей.", []),
    "sleep": ("Тимур плохо спит после футбола.", ["Тимур"]),
}

TWO = "что Злате нужно обязательно взять с собой на плавание в четверг и какой чай она любит?"
OTHER_TEA = {"timur_tea", "ira_tea", "vera_tea", "gleb_tea", "tea_shop"}


def _store(tmp_path, on: bool, names: bool = True, extra: dict | None = None, **config):
    config = {"db_path": str(tmp_path / ("two-on.db" if on else "two.db")), "hrr_dim": 1024, **config}
    if on:
        config["prefetch_clause_coverage"] = True
    provider = HolographicMemoryProvider(config=config)
    provider.initialize(session_id="two")
    back = {}
    for key, (text, entities) in {**FACTS, **(extra or {})}.items():
        args = {"action": "add", "content": text}
        if names and entities:
            args["entities"] = entities
        back[json.loads(provider.handle_tool_call("fact_store", args))["fact_id"]] = key
    return provider, back


@pytest.fixture
def on(tmp_path):
    provider, back = _store(tmp_path, True)
    yield provider, back
    provider.shutdown()


@pytest.fixture
def off(tmp_path):
    provider, back = _store(tmp_path, False)
    yield provider, back
    provider.shutdown()


def _recalled(provider, back, query, turn=0) -> list[str]:
    provider.on_session_switch(f"t{turn}", reset=True)
    block = provider.prefetch(query)
    return [back[int(line.split()[1][1:])] for line in block.splitlines()[1:]] if block else []


# -- the case ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("query, targets", [
    (TWO, {"zlata_swim", "zlata_tea"}),
    ("Что Злате нужно обязательно взять с собой на плавание в четверг? И какой чай она любит?",
     {"zlata_swim", "zlata_tea"}),
    ("напомни, что Тимуру нужно взять с собой на футбол в субботу, и какой сок он любит",
     {"timur_football", "timur_juice"}),
])
def test_both_questions_are_answered_in_the_top_five(on, query, targets):
    provider, back = on
    recalled = _recalled(provider, back, query)
    assert targets <= set(recalled[:5]), recalled


def test_the_second_question_does_not_bring_another_persons_answer(on):
    provider, back = on
    recalled = _recalled(provider, back, TWO)
    assert not OTHER_TEA & set(recalled), recalled


def test_a_rescued_fact_comes_after_the_facts_the_gate_kept(on):
    provider, back = on
    rows = {key: provider._retriever._rows([fact_id])[0] for fact_id, key in back.items()}
    # tea first in the pool: the gate keeps the Thursday fact, the rescue may only add after it
    pool = [rows["zlata_tea"], rows["zlata_swim"], rows["timur_tea"]]
    kept = [back[f["fact_id"]] for f in provider._retriever.relevant(TWO, pool, provider._min_coverage)]
    assert kept == ["zlata_swim", "zlata_tea"]


# -- what stays as it was ------------------------------------------------------------------------------------------

def test_off_by_default(off):
    provider, back = off  # no prefetch_clause_coverage in the config: the gate of 0.22.1, tea left out
    recalled = _recalled(provider, back, TWO)
    assert "zlata_swim" in recalled and "zlata_tea" not in recalled, recalled


@pytest.mark.parametrize("query", [
    "привет, как дела? и как погода?",
    "ну и денёк, а как у тебя?",
    "я выпью чай и пойду спать",
    "ну что, как ты там и что у тебя нового?",
    "что-то я устал, и какой смысл спорить?",
])
def test_chit_chat_of_two_questions_recalls_nothing(on, query):
    provider, back = on
    assert _recalled(provider, back, query) == []


def test_a_question_about_someone_the_message_does_not_name_is_not_guessed(tmp_path):
    # «ире» typed in lower case and the store declared no names: whose tea «она» likes cannot be told, so none of the
    # tea facts is let in (each of them holds «чай» and «любит»)
    provider, back = _store(tmp_path, True, names=False)
    try:
        recalled = _recalled(provider, back, "что ире нужно взять с собой в среду и какой чай она любит?")
    finally:
        provider.shutdown()
    assert not {"zlata_tea", "timur_tea", "ira_tea", "vera_tea", "gleb_tea"} & set(recalled), recalled


def test_a_second_question_sharing_one_word_with_a_fact_does_not_bring_it(on):
    provider, back = on  # every tea fact holds «чай» and nothing else of «какой чай заварить гостям»
    recalled = _recalled(provider, back,
                         "что Злате нужно обязательно взять с собой на плавание в четверг и какой чай заварить гостям?")
    assert not (OTHER_TEA | {"zlata_tea"}) & set(recalled), recalled


@pytest.mark.parametrize("message, parts", [
    (TWO, ("что Злате нужно обязательно взять с собой на плавание в четверг", "какой чай она любит?")),
    ("Что Злате взять в четверг? И какой чай она любит?", ("Что Злате взять в четверг", "какой чай она любит?")),
    ("напомни, что Тимуру взять в субботу, и во сколько у него футбол",
     ("напомни, что Тимуру взять в субботу", "во сколько у него футбол")),
    ("What should Dina take on Tuesday and which tea does she like?",
     ("What should Dina take on Tuesday", "which tea does she like?")),
    ("какой чай и кофе она любит?", None),  # «и» between two words of one question
    ("Злата любит чай и как это понимать", None),  # the first part asks nothing
    ("Что взять? Какой чай? Когда день рождения?", None),  # three questions: not two
    ("я выпью чай и пойду спать", None),
    # one question about two things: "which tea" alone asks nothing, the verb is in the second half (Codex review)
    ("which tea and which coffee does she like?", None),
    ("какой зелёный чай и какой кофе она любит?", None),
    # a quoted line is what is asked about, not two questions of the message (Codex review)
    ("Что означает у Златы длинная надпись про плавание очки шапочку и четверг „Что упаковать? Какой чай она любит?“?",
     None),
    ("Что значит надпись «что взять в четверг и какой чай она любит»?", None),
])
def test_the_split_finds_exactly_two_questions(message, parts):
    from hermes_plugin_holographic.text import two_questions  # new in this experiment: the other tests run on 0.22.1
    assert two_questions(message) == parts


# -- whose answer it is (Codex review 2026-10-29, finding 1) --------------------------------------------------------

def _on_and_off(tmp_path, query, extra=None) -> tuple[list, list]:
    """What the message recalls with prefetch_clause_coverage on and off, over the same facts."""
    out = []
    for on in (True, False):
        provider, back = _store(tmp_path, on, extra=extra)
        try:
            out.append(_recalled(provider, back, query))
        finally:
            provider.shutdown()
    return out[0], out[1]


def test_she_of_one_of_two_people_brings_neither_answer(tmp_path):
    # «она» may be Злата or Тимур: whose tea is asked cannot be told, so no tea fact is let in, Тимур's least of all
    on, off = _on_and_off(tmp_path, "что Тимуру и Злате нужно обязательно взять с собой на плавание в четверг и какой "
                                    "чай она любит?")
    assert "timur_tea" not in on, on
    assert on == off, (on, off)


def test_a_name_the_store_does_not_know_brings_nobodys_answer(tmp_path):
    # «клара» typed in lower case is a word no fact holds: it may be a name, and the tea of Ира or Тимур is not hers
    on, off = _on_and_off(tmp_path, "что Злате нужно обязательно взять с собой на плавание в четверг и какой чай любит "
                                    "клара?")
    assert not OTHER_TEA & set(on), on
    assert on == off, (on, off)


def test_a_word_that_is_also_a_known_name_does_not_say_whose_answer_it_is(tmp_path):
    # a dog called «Чай»: the word «чай» in the question is no name of anybody, and «она» stays Злата's or nobody's
    dog = {"dog": ("Чай — так зовут собаку Глеба.", ["Чай"])}
    on, off = _on_and_off(tmp_path, TWO, extra=dog)
    assert not OTHER_TEA & set(on), on


def test_a_quoted_question_is_not_split(tmp_path):
    # the message asks what an inscription means; the questions are the inscription's (Codex review, finding 2)
    on, off = _on_and_off(tmp_path, "Что означает у Златы длинная надпись про плавание очки шапочку и четверг „Что "
                                    "упаковать? Какой чай она любит?“?")
    assert on == off, (on, off)


def test_chit_chat_that_recalls_on_main_recalls_the_same(tmp_path):
    # on 0.22.1 already «неделю» brings «Орхидеи поливать раз в неделю» (one rare word of three): a weakness of the
    # gate, not of this experiment, which must not change it either way
    on, off = _on_and_off(tmp_path, "как прошла неделя и что нового?")
    assert on == off, (on, off)


# -- the reranker (finding 3) ---------------------------------------------------------------------------------------

class _BergamotFirst:
    """rerank._OPENER, the endpoint: ranks the document about bergamot tea first, the others as sent."""

    def __init__(self):
        self.calls = 0

    def open(self, request, timeout=None):
        self.calls += 1
        documents = json.loads(request.data)["documents"]
        order = sorted(range(len(documents)), key=lambda i: (0 if "бергамот" in documents[i] else 1, i))
        results = [{"index": i, "relevance_score": float(len(order) - pos)} for pos, i in enumerate(order)]
        return _Answer(json.dumps({"results": results}).encode())


class _Answer:
    def __init__(self, body: bytes):
        self.body, self.at = body, 0

    def read(self, n=-1):
        piece = self.body[self.at:] if n is None or n < 0 else self.body[self.at:self.at + n]
        self.at += len(piece)
        return piece

    read1 = read

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


SWIM = {  # more answers to the first question: the gate keeps enough facts to fill the five places
    "swim_towel": ("В четверг Злате взять на плавание полотенце.", ["Злата"]),
    "swim_goggles": ("Злате на плавание в четверг взять новые очки.", ["Злата"]),
    "swim_coach": ("На плавание в четверг Злате взять с собой справку для тренера.", ["Злата"]),
    "swim_cap": ("Шапочку для плавания Злате взять с собой в четверг.", ["Злата"]),
}


def test_a_rescued_fact_ranked_first_takes_no_place_of_a_kept_one(tmp_path, monkeypatch):
    endpoint = _BergamotFirst()
    monkeypatch.setattr(rerank, "_OPENER", endpoint)
    blocks = []
    for on in (True, False):
        provider, back = _store(tmp_path, on, extra=SWIM, rerank="on", rerank_url="https://rerank.example.net/v1/rerank",
                                rerank_model="fake-reranker", rerank_window="0.0")
        try:
            blocks.append(_recalled(provider, back, TWO))
        finally:
            provider.shutdown()
    on, off = blocks
    assert endpoint.calls >= 2, "the reranker was not asked"
    assert len(off) == 5, off  # five facts without the rescue: no place is free
    assert set(off) <= set(on), (on, off)  # none of them gives its place to the rescued tea


# -- the cost of the split (finding 6) ------------------------------------------------------------------------------

def test_the_split_is_linear_in_a_long_run_of_spaces():
    # a pasted message with a long run of spaces: a regular expression that looks ahead from every space took about
    # 0.35 s for 4 000 of them and grows with the square; one pass takes milliseconds. The bound is generous.
    from hermes_plugin_holographic.text import two_questions
    message = "what" + " " * 20_000 + "nothing"
    start = time.perf_counter()
    assert two_questions(message) is None
    assert time.perf_counter() - start < 1.0
