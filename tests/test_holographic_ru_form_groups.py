"""Case forms of Russian nouns that share no key (text._RU_FORM_GROUPS: an experiment, off by default).

Snowball cuts «чай» to «ча», «мае» to «ма», «дня» to «дн»; a stem under three letters is no key (as a prefix «ча*»
would find «час» and «чат»), so «чая» never met «чай» and «какие планы на май?» never found «В мае едем на дачу». A
dropped vowel does the same to longer words: «подарки» / «подарок», «отца» / «отец». One key for every two-letter
stem was rejected: it joins different words («ель» / «ели» of «есть», «рой» / «рою» of «рыть», «уха» / «ухо»). The
experiment lists a few common nouns with their own case forms, each form checked, and a word meets another only
through its own noun's list. The facts are synthetic."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic import text

FACTS = {
    "may": "В мае едем на дачу.",
    "tea_shop": "Чай покупаем в лавке на углу.",
    "gift": "Подарок для Тимура спрятан в шкафу.",
    "dad_car": "Машина отца стоит у подъезда.",
    "fir": "В лесу у дома растёт старая ель.",
    "soup": "На ужин сварить уху из форели.",
    "bees": "Пчелиный рой улетел в сад.",
    "ear": "У Тимура болит ухо после бассейна.",
    "plants": "Орхидеи поливать раз в неделю.",
    "school": "Школа начинается в восемь утра.",
    "car": "Машину заправлять на станции у рынка.",
}


def _clear():
    text._keys.cache_clear()
    text.analyze.cache_clear()
    if hasattr(text, "_ru_form_keys"):
        text._ru_form_keys.cache_clear()


@pytest.fixture
def groups(monkeypatch):
    # raising=False: on 0.22.1 the switch does not exist, setting it changes nothing, and these tests show its behaviour
    monkeypatch.setattr(text, "_RU_FORM_GROUPS", True, raising=False)
    _clear()
    yield
    monkeypatch.undo()
    _clear()


def _store(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "forms.db"), "hrr_dim": 1024})
    provider.initialize(session_id="forms")
    back = {}
    for key, fact in FACTS.items():
        back[json.loads(provider.handle_tool_call("fact_store", {"action": "add", "content": fact}))["fact_id"]] = key
    return provider, back


@pytest.fixture
def store(tmp_path):
    provider, back = _store(tmp_path)
    yield provider, back
    provider.shutdown()


def _recalled(provider, back, query, turn=0) -> set:
    provider.on_session_switch(f"t{turn}", reset=True)
    block = provider.prefetch(query)
    return {back[int(line.split()[1][1:])] for line in block.splitlines()[1:]} if block else set()


@pytest.mark.parametrize("query, wanted", [
    ("Какие планы на май?", "may"),  # «май» / «мае»: the stem «ма»
    ("Какие чаи у нас есть?", "tea_shop"),  # «чаи» / «чай»: «ча»; the question's only word
    ("Какие подарки уже купили?", "gift"),  # «подарки» / «подарок»: «подарк» / «подарок»
    ("Где стоит машина у отца?", "dad_car"),  # «отца» in both: a control, found without the lists too
])
def test_a_case_form_finds_a_fact_with_another_form_of_the_noun(groups, store, query, wanted):
    provider, back = store
    assert wanted in _recalled(provider, back, query)


@pytest.mark.parametrize("query, key", [
    ("Что вы ели вчера вечером?", "fir"),  # «ели» of «есть» is no form of «ель» in the list
    ("Я рою грядку у забора?", "bees"),  # «рою» of «рыть», «рой» is not listed at all
    ("Болит ухо у Тимура?", "soup"),  # «ухо» and «уха» are two nouns: «уху» is in neither list
    ("Где сварить уху?", "ear"),
])
def test_other_words_with_the_same_short_stem_stay_apart(groups, store, query, key):
    provider, back = store
    assert key not in _recalled(provider, back, query)


def test_the_conflict_check_still_sees_a_negation_in_another_case_form(groups, store):
    provider, _ = store  # the forms' keys also reach the write-time checks: a negation stays a possible conflict
    add = lambda fact: json.loads(provider.handle_tool_call("fact_store", {"action": "add", "content": fact}))  # noqa: E731
    add("Злата любит чай с бергамотом.")
    result = add("Злата не любит чая с бергамотом.")
    assert [c["content"] for c in result.get("possible_conflicts", ())] == ["Злата любит чай с бергамотом."]


def test_off_by_default(store):
    provider, back = store  # the switch untouched: «мае» does not meet «май», as in 0.22.1
    assert not getattr(text, "_RU_FORM_GROUPS", False)
    assert "may" not in _recalled(provider, back, "Какие планы на май?")


def test_a_form_written_with_e_for_yo_is_the_same_form(groups):
    assert text.analyze("ребенка").words[0].keys & text.analyze("ребёнок").words[0].keys
    assert text.analyze("подарки").words[0].keys & text.analyze("подарок").words[0].keys


def test_a_light_word_is_not_listed(groups):
    # «дни рождения» would meet every «День рождения …»: «день» keeps its own keys (measured, see text._RU_FORMS)
    assert text.analyze("дни").words[0].keys.isdisjoint(text.analyze("день").words[0].keys)


# -- Codex review 2026-10-29 ----------------------------------------------------------------------------------------

def test_the_forms_do_not_make_a_new_fact_a_duplicate(groups, store):
    # the month «мае» and the name «Мая» are not one word to the duplicate check: auto-extraction skips a fact that
    # near_duplicate finds, and this one says something else (finding 4)
    provider, _ = store
    provider.handle_tool_call("fact_store", {"action": "add", "content": "Мая предпочитает чай с бергамотом."})
    assert provider._retriever.near_duplicate("В мае я предпочитаю чай с бергамотом.") is None


@pytest.mark.parametrize("form, noun", [
    ("матери", "мать"),  # also «матери!», the imperative of «материть»
    ("потолок", "потолка"),  # also the past tense of «потолочь»
])
def test_a_form_that_is_also_a_verb_joins_no_group(groups, form, noun):
    assert text.analyze(form).keys.isdisjoint(text.analyze(noun).keys)


@pytest.mark.parametrize("fact, query", [
    ("Я маю новий план.", "май"),  # Ukrainian «маю», "I have" (finding 5, Codex's case)
    ("Мая сястра — гэта Ганна.", "Что будет в мае?"),  # Belarusian «мая», "my": no letter Russian lacks
])
def test_text_in_another_language_of_cyrillic_script_gets_no_russian_forms(groups, tmp_path, fact, query):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "uk.db"), "hrr_dim": 1024})
    provider.initialize(session_id="uk")
    try:
        provider.handle_tool_call("fact_store", {"action": "add", "content": fact})
        provider.handle_tool_call("fact_store", {"action": "add", "content": "Школа начинается в восемь утра."})
        provider.on_session_switch("t0", reset=True)
        assert fact not in provider.prefetch(query)
    finally:
        provider.shutdown()
