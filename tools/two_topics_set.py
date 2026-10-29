# -*- coding: utf-8 -*-
"""Two-topic set — the measurement stand for two experiments, both off by default: prefetch_clause_coverage (a message
of two questions: retrieval.FactRetriever._clause_rescue) and text._RU_FORM_GROUPS (case forms of Russian nouns that
share no key: «чай» / «чая», «май» / «мае», «подарок» / «подарки»).

The existing sets have no message that asks two separate things (household's multi-target questions ask one thing about
two people), so this set adds them: a Russian household of 34 facts where several people like tea, juice or coffee, so
that the second question has answers about the wrong person (`confusers`). Slices: t — two questions in one message;
c — one question (controls); s — a case form in the question, another in the fact; CHITCHAT — must recall nothing.
Synthetic data: the household is invented, no real people.

    PYTHONPATH=<hermes-agent> python tools/two_topics_set.py --variants off,clause,forms,both --out <dir>

Runs each variant over this set, contrast_ru, household (en/ru/zh) and the multilingual set (8 languages) with rerank
and embeddings off, writes rows_<variant>.jsonl and prints, against the first variant: hits@5, all_targets@5 (questions
with two or more labels), noise, confusion, every target lost or gained in the top five and every row that changed.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

# key -> (text, declared names)
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
    "tea_tin": ("Банка с чаем стоит на верхней полке.", []),
    "trash": ("Мусор вывозят во вторник и в пятницу.", []),
    "school": ("Школа у детей начинается в восемь утра.", []),
    "umbrella": ("В сезон дождей брать с собой зонт.", []),
    "market": ("Рынок работает по выходным до двух часов.", []),
    "dentist": ("Стоматолог принимает по средам после обеда.", []),
    "plants": ("Орхидеи поливать раз в неделю.", []),
    "car": ("Машину заправлять на станции у рынка.", []),
    "pool_bag": ("Сумку для бассейна хранить в прихожей.", []),
    "honey": ("Мёд берём у пасечника на рынке.", []),
    "day_off": ("В праздничный день рынок не работает.", []),
    "sleep": ("Тимур плохо спит после футбола.", ["Тимур"]),
    "may": ("В мае едем на дачу.", []),
}

_TEA = ["timur_tea", "ira_tea", "vera_tea", "gleb_tea"]
# (qid, question, targets, confusers): every target must be in the top five; a confuser there is a confusion
QUERIES = [
    ("t01", "что Злате нужно взять с собой в четверг и какой чай она любит?", ["zlata_swim", "zlata_tea"], _TEA),
    ("t02", "что Злате нужно обязательно взять с собой на плавание в четверг и какой чай она любит?",
     ["zlata_swim", "zlata_tea"], _TEA),
    ("t03", "что Тимуру нужно взять с собой в субботу и какой сок он любит?", ["timur_football", "timur_juice"], []),
    ("t04", "напомни, что Тимуру нужно взять с собой на футбол в субботу, и какой сок он любит",
     ["timur_football", "timur_juice"], []),
    ("t05", "что Ире нужно взять с собой в среду и какой чай она любит?", ["ira_yoga", "ira_tea"],
     ["timur_tea", "zlata_tea", "vera_tea", "gleb_tea"]),
    ("t06", "что Злате нужно взять с собой в четверг и какой чай она пьёт?", ["zlata_swim", "zlata_tea"], _TEA),
    ("t07", "во сколько у Тимура футбол в субботу и какой чай он любит?", ["timur_football", "timur_tea"],
     ["zlata_tea", "ira_tea", "vera_tea", "gleb_tea"]),
    ("t08", "когда у Златы день рождения и на что у неё аллергия?", ["zlata_bd", "zlata_allergy"], ["timur_bd"]),
    ("t09", "какой корм покупать Мусе и где хранить сумку для бассейна?", ["musya_food", "pool_bag"], []),
    ("t10", "когда вывозят мусор и где мы покупаем чай?", ["trash", "tea_shop"], []),
    ("t11", "во сколько начинается школа у детей и где хранить сумку для бассейна?", ["school", "pool_bag"], []),
    ("t12", "какие таблетки нужно пить бабушке Нине утром и какой корм покупать Мусе?", ["nina_pills", "musya_food"],
     []),
    ("t13", "что Ире взять с собой в среду и какой чай любит Злата?", ["ira_yoga", "zlata_tea"],
     ["timur_tea", "vera_tea", "gleb_tea"]),
    ("t14", "Что Злате нужно обязательно взять с собой на плавание в четверг? И какой чай она любит?",
     ["zlata_swim", "zlata_tea"], _TEA),
    ("t15", "где работает Вера до шести вечера и что она любит читать перед сном?", ["vera_work", "vera_books"],
     ["timur_juice", "zlata_tea"]),
    ("t16", "что Глеб пьёт по утрам с молоком и где он работает по пятницам?", ["gleb_coffee", "gleb_work"],
     ["vera_tea", "vera_work"]),
    ("c01", "какой чай любит Злата?", ["zlata_tea"], _TEA),
    ("c02", "что Злате взять с собой в четверг?", ["zlata_swim"], []),
    ("c03", "какой сок любит Тимур?", ["timur_juice"], ["timur_tea"]),
    ("s01", "какие планы на май?", ["may"], []),
    ("s02", "какие чаи у нас есть?", ["tea_shop", "tea_tin"], []),
    ("s03", "где банка с чаем?", ["tea_tin"], []),
]
CHITCHAT = [
    "привет, как дела? и как погода?",
    "ну и денёк, а как у тебя?",
    "я выпью чай и пойду спать",
    "ну что, как ты там и что у тебя нового?",
    "что-то я устал, и какой смысл спорить?",
    "спасибо, а что ещё скажешь?",
    "ладно, а ты как думаешь?",
    "хорошо, и что дальше?",
]

_REPO = Path(__file__).resolve().parent.parent


def _package():
    agent = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
    sys.path.insert(0, str(agent))
    if "hermes_plugin_holographic" not in sys.modules:
        spec = importlib.util.spec_from_file_location("hermes_plugin_holographic", _REPO / "__init__.py",
                                                      submodule_search_locations=[str(_REPO)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["hermes_plugin_holographic"] = module
        spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location("provenance_probe", _REPO / "tools" / "provenance_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    return sys.modules["hermes_plugin_holographic"], probe


def _switch(text, forms: bool) -> None:
    text._RU_FORM_GROUPS = forms
    for cached in (text._keys, text.analyze, text._ru_form_keys):
        cached.cache_clear()


def run_variant(variant: str, out: Path) -> list[dict]:
    pkg, probe = _package()
    from hermes_plugin_holographic import text
    _switch(text, variant in ("forms", "both"))
    extra = {"prefetch_clause_coverage": True} if variant in ("clause", "both") else None
    work = out / f"tmp_{variant}"
    work.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(work)
    try:
        rows = probe.run_contrast("contrast_ru", ["off"], "off", "", extra=extra)
        rows += probe.run_household(["off"], "off", "", split="all", extra=extra)
        rows += probe.run_multilingual(["en", "ru", "zh", "de", "ja", "it", "nl", "th"], ["off"], "off", "", extra=extra)
        p = pkg.HolographicMemoryProvider(config={"db_path": str(work / "two_topics.db"), "hrr_dim": 1024,
                                                  **(extra or {})})
        p.initialize(session_id="two")
        back = {}
        for key, (fact, names) in FACTS.items():
            args = {"action": "add", "content": fact, **({"entities": names} if names else {})}
            back[int(json.loads(p.handle_tool_call("fact_store", args))["fact_id"])] = key
        for qid, query, targets, confusers in QUERIES:
            p.on_session_switch(f"two_{qid}", reset=True)
            keys = [back[f] for f in probe.parse_ids(p.prefetch(query)) if f in back]
            rows.append({"set": "two_topics", "lang": "ru", "qid": qid, "kind": "real", "q": query, "keys": keys,
                         "targets": targets, "confused": sorted(set(keys[:5]) & set(confusers))})
        for i, query in enumerate(CHITCHAT):
            p.on_session_switch(f"two_chit{i}", reset=True)
            keys = [back[f] for f in probe.parse_ids(p.prefetch(query)) if f in back]
            rows.append({"set": "two_topics", "lang": "ru", "qid": f"chit{i}", "kind": "chit", "q": query,
                         "keys": keys, "targets": [], "confused": []})
        p.shutdown()
    finally:
        tempfile.tempdir = None
        _switch(text, False)
    keep = ("set", "lang", "qid", "kind", "q", "keys", "targets", "confused", "wrong", "wrong5")
    with open(out / f"rows_{variant}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + "\n")
    return rows


def _confused(r: dict) -> bool:
    if r["set"] in ("household", "two_topics"):
        return bool(r.get("confused"))
    return bool(r.get("wrong5")) if r["set"] == "multi" else bool(r.get("wrong"))


def compare(out: Path, variants: list[str]) -> None:
    data = {v: {(r["set"], r["lang"], r["qid"]): r for r in map(json.loads, open(out / f"rows_{v}.jsonl",
                                                                                   encoding="utf-8"))}
            for v in variants}
    base = data[variants[0]]
    groups = collections.defaultdict(list)
    for key in base:
        groups[(key[0], key[1])].append(key)
    print(f"{'set':<12} {'lang':<12}" + "".join(f" | {v}: hit5 all5 noise conf changed" for v in variants))
    for g in sorted(groups):
        line = f"{g[0]:<12} {g[1]:<12}"
        for v in variants:
            rows = [data[v][k] for k in groups[g] if k in data[v]]
            real = [r for r in rows if r["kind"] == "real" and r["targets"]]
            multi = [r for r in real if len(r["targets"]) > 1]
            noise = [r for r in rows if r["kind"] == "chit" or not r["targets"]]
            line += (f" | {sum(bool(set(r['targets']) & set(r['keys'][:5])) for r in real)}/{len(real)}"
                     f" {sum(set(r['targets']) <= set(r['keys'][:5]) for r in multi)}/{len(multi)}"
                     f" {sum(bool(r['keys']) for r in noise)}/{len(noise)}"
                     f" {sum(_confused(r) for r in real)}"
                     f" {sum(1 for k in groups[g] if k in data[v] and data[v][k]['keys'] != base[k]['keys'])}")
        print(line)
    for v in variants[1:]:
        lost, gained = [], []
        for k, r in base.items():
            cur = data[v].get(k)
            if cur is not None:
                a, b = set(r["targets"]) & set(r["keys"][:5]), set(cur["targets"]) & set(cur["keys"][:5])
                lost += [f"{'/'.join(k)}:{t}" for t in sorted(a - b)]
                gained += [f"{'/'.join(k)}:{t}" for t in sorted(b - a)]
        print(f"\n{v} vs {variants[0]}: lost {len(lost)} {lost}; gained {len(gained)} {gained}")
        for k, r in base.items():
            cur = data[v].get(k)
            if cur is not None and cur["keys"] != r["keys"]:
                print(f"  {'/'.join(k)} {r['q']!r}\n    {variants[0]}: {r['keys'][:6]}\n    {v}: {cur['keys'][:6]}"
                      f"  (targets {r['targets']})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", default="off,clause,forms,both")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        run_variant(v, out)
    compare(out, variants)


if __name__ == "__main__":
    main()
