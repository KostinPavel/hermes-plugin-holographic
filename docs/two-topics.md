# Two questions in one message (experimental)

> **Experimental.** `prefetch_clause_coverage` is off by default and not offered by the setup wizard. It was measured
> only on a synthetic set written for it; it has not been checked on real conversations or with a reranker on real
> data. The way it decides, and the key itself, may change or go away in a later release. Turn it on to try it, and
> read the risks below first.

## What it does

Before each turn the provider recalls up to five facts. A fact gets in only when it covers enough of the message:
`prefetch_min_coverage` (0.3 by default) of the weight of the message's informative words, rarer words weighing more.
A message that asks two things at once gives each answer only its half of the words:

> что Злате нужно обязательно взять с собой на плавание в четверг и какой чай она любит?

The fact «Злата любит чай с бергамотом.» holds every word of the second question (Злате, чай, любит), and still covers
only 0.24 of the whole message, under 0.3. The fact about Thursday swimming is recalled; the one about tea is not.

With `prefetch_clause_coverage` on, a fact the coverage gate left out gets a second look against each of the message's
two questions alone, with the same threshold. It is let in only when all of this holds:

- the message splits into **exactly two questions** that each stand alone (below);
- the fact holds **at least two words of that question exactly**, and covers `prefetch_min_coverage` of the question's
  own words; a word of the question that no fact holds counts against it, as in the gate;
- **whose answer it is can be told**: a question that names a person (with a capital inside the sentence) is bound to
  exactly one entity the store knows; «она» / «он» / "she" / "he" is bound to the one person the other question names.
  The fact must then name that person itself, with a capital.

A rescued fact comes after every fact the gate kept. With the reranker on, it only takes a place among the five that the
other facts leave free: the reranker may rank it first, but it never pushes out a fact that would have been recalled
without this option. Only the recall before a turn (prefetch) changes; `fact_store search` is unchanged.

## Turning it on

In `config.yaml` under `plugins.hermes-memory-store`, then restart the gateway or start a new CLI session:

```yaml
prefetch_clause_coverage: on   # true / yes / 1 work too; default off
```

Unlike the other keys, a value it does not recognise (`maybe`, `onn`) is read as off **without a warning**: memory
keeps working, the option is simply not on. Check the spelling if nothing changes.

## How a message is split

Only Russian and English question words are recognised (что, какой, где, когда, сколько, кто, почему…; what, which,
where, when, who, why, how…), also after a preposition («во сколько», "at what"). The second question starts at a
question word after «?» or «;», or after «и» / «а» / «а также» / "and" (with a comma or not). The first part must have
a question word too, and each part must stand alone: «какой чай и какой кофе она любит?» and "which tea and which coffee
does she like?" are one question about two things and are not split. A boundary inside quotes does not count, and an
unclosed quote is not split. Three questions or more — not split. When in doubt the message is left whole and recall
is exactly as with the option off. The split is one linear pass over the message.

## When it helps

A message with two separate questions where one answer is short and the other half of the message is long, and the
person in the second question is named or is «она» / «он» of a person named in the first:

- «Что Злате нужно обязательно взять с собой на плавание в четверг? И какой чай она любит?»
- «напомни, что Тимуру нужно взять с собой на футбол в субботу, и какой сок он любит»
- "What should Dina take on Tuesday and which tea does she like?"

If the gate already keeps both answers (short messages often pass at 0.3 anyway), nothing changes.

## What it does not catch

Each of these is left as it was with the option off — no rescue, and no wrong answer either:

- **A different word in the fact**: «…и какой чай она **пьёт**?» when the fact says «Злата **любит** чай»: from the
  second question only «чай» matches, and one word is not enough.
- **A person the store does not know as an entity**: a name that was never declared or guessed (a short name such as
  «Ире» in a store without the entity «Ира»), or a name typed in lower case («какой чай любит клара?»): whose answer it
  is cannot be told, so nothing is let in.
- **«она» of one of two people**: «что Тимуру и Злате нужно взять… и какой чай она любит?».
- **A question in quotes**: «Что значит надпись „что взять в четверг и какой чай она любит“?» — it asks about the
  inscription, so it is not split.
- **A question without a question word at the start of its part**: «Злате в четверг что взять, и чай какой любит».
- **One question about two things** («какой чай и кофе она любит?»), and three questions or more.
- **Other languages**: only Russian and English are split; Ukrainian and Belarusian question words («що», «який»,
  «што», «які») are not in the list, so such messages are not split. Chinese and Japanese text is not split either.
- **A fact the search did not bring as a candidate**: the rescue only looks again at the candidates of this turn that
  the gate rejected; it does not search again.

## Risks

- **Another person's answer.** The rescue checks the name, not the meaning. A fact that names the right person and
  shares two words with the question but answers something else can come in, and so can a fact that names that person
  together with someone else. The guards (a known entity, a capital, exactly one person, two exact words) are tested on
  the synthetic household; real stores have more people and more facts per person.
- **Noise.** It only adds facts, so a turn can get one or two lines that do not help, and they cost tokens. Chit-chat in
  two parts («привет, как дела? и как погода?») recalls nothing on the test set, but real chat was not measured.
- **A name that is also a word.** When a known entity is called by an ordinary word (a dog called «Чай»), the word «чай»
  in a question is no longer proof of anything: such a question binds to nobody and nothing is rescued. Safe, but in
  such a household the option helps less.
- **Ukrainian and Belarusian.** The question words are Russian and English, so a Ukrainian or Belarusian message is
  normally not split. Nothing here proves that a message is Russian; it is just not recognised as two questions.
- **Reranker.** With `rerank: on` the rescued facts join the pool that is sent to the endpoint with the others (one or
  two more candidates for the same call) and only take free places in the top five. This combination is covered by a
  test with a fake endpoint; it has not been measured with a real reranker on real data.
- **Latency.** On a message the gate rejected a candidate for, the message is analysed once more and each question's
  person is looked up; this runs under the provider lock. On the 30-fact synthetic store it added about 2 ms to the
  median prefetch on a slow laptop (i5-1135G7); a large store was not measured. Messages that do not split cost one
  linear pass.
- **Only Russian and English** questions are handled at all.

## Numbers

Stand: `tools/two_topics_set.py` — a synthetic Russian household (invented people; several of them like tea, juice or
coffee, so the second question has answers about the wrong person), 16 messages with two questions, controls, case-form
questions and chit-chat; plus the existing sets (contrast_ru, household en/ru/zh, the multilingual set in 8 languages,
about 620 questions). Rerank and search by meaning off. Measured again on the current code (after the fixes of the Codex
review of 2026-10-29), with the same result as the first run.

| | off | on |
|---|---|---|
| Both answers in the top five, 16 two-question messages | 12 of 16 | 15 of 16 |
| Changed answers on the existing sets (~620 questions) | — | 0 |
| Answers lost anywhere | — | 0 |

The three gained: the tea of «Злата» in two wordings (one sentence with «и», two sentences with «? И») and the juice
of «Тимур». The one still missing is «…и какой чай она пьёт?» (another verb in the fact, see above). The stand's own
table prints 12/17 → 15/17: it also counts a case-form question with two answers («какие чаи у нас есть?»), missed either
way. The synthetic set was written together with the feature, so it shows that it works and that it is safe on the older sets, not how often it helps in
real conversation. **On live data it did not help**: a read-only run over one live install's history (5,081 human
messages, its own store) split 73 messages into two questions and let in 0 facts the gate had left out, displaced none,
and cost +2-16 ms at p95. So that install keeps it off, and it stays an experiment until a store where two-question
messages miss facts shows it rescuing them.

Rerun: `PYTHONPATH=<hermes-agent> python tools/two_topics_set.py --variants off,clause --out <dir>`.

## Case forms of short Russian nouns: not a feature

The same branch carries a second experiment: hand-written case forms of about thirty short Russian nouns («чай / чая /
чаи», «май / мае», «подарок / подарки», «отец / отца»), whose stems Snowball cuts too short to match. It is **not a
setting and not supported**: it is a switch inside the code (`text._RU_FORM_GROUPS = False`), off, with no config key,
and it is not planned to be released in this form.

Why not: the list is hand-made and short; homonyms were removed by hand without a full dictionary of verbs («ели» of
«есть», «маю», «матери» are out, others may remain); a word that becomes findable also exposes an old weakness of the
gate on chit-chat («В мае бывает дождливо.» started to recall a fact about a holiday in May); and «Russian text» is
decided only by the absence of Ukrainian or Belarusian letters and common words — the absence of signs of another
language, not proof of Russian. Its keys are used for search and recall only; duplicate and conflict checks at write
time ignore them. The way forward, if any, is a table built from a full dictionary of word forms (a licence decision
first), not a longer hand list.
