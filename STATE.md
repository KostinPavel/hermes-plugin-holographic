# STATE — состояние на 2026-10-22 (handoff для продолжения на другом ПК)

Клон/рабочая копия: ветка `feature/rerank-benchmark`, PR **#10** в
`itpartypattaya/hermes-plugin-holographic` (draft; head = этот коммит).
Origin форка: `KostinPavel/hermes-plugin-holographic`. Всё суточное — в git;
этот файл и артефакты в `results/` — чтобы не терять контекст между ПК.

## Инфраструктура (та же, что на этом ПК)

- Python: `/home/pavel/.hermes/hermes-agent/venv/bin/python`,
  `PYTHONPATH=/home/pavel/.hermes/hermes-agent`,
  `HERMES_AGENT_REPO=/home/pavel/.hermes/hermes-agent`.
- Эндпоинты (systemd user-сервисы, Linger): реранк `http://127.0.0.1:1235/v1/rerank`
  (bge-m3-reranker Q4_K_M, GTX 1050 Ti 4GB), эмбеддинги `http://127.0.0.1:1237/v1/embeddings`
  (bge-m3 Q4). Прогоны только на GPU.
- Тесты: `PYTHONPATH=... python -m pytest tests/ -q` → **2490 passed**; `ruff check` чист.
- Прогоны свипов: `python -I tools/<probe>.py` из корня репо (bootstrap sys.path внутри).

## Сделано за 2026-10-22

1. **Фикс `catch_up_vectors` (`b4674cd`) — важно понимать при чтении старых цифр.**
   Проба проверяла `ready()` («есть хоть один вектор», `LIMIT 1`), а бюджет
   `_catch_up_seconds` (8 с) отсекал хвост pending: на wide-базе (637 фактов)
   не встраивались fact_id 1..25 — цели real-запросов. Все wide-цифры до фикса
   невалидны (симптом: cmax real 0.91→0.55 у 70/90 запросов). Фикс: цикл по
   `idx.pending(BATCH, allowed=_sendable)`. Проверка: 637/637, brute == meaning.
2. **Wide-A/B пересчитан** (`9a7a7a3`, PROVENANCE §Wide): off 78=78 (ретривал не
   страдает), gate 77→72, шум 80→86, wrong5 29→19 off / 6→4 gate, pool p50 21-24,
   p95 400-530 мс, deep-lift 0 подтверждён на чистом индексе.
3. **Этап-0 admission — три свипа, ждёт решения пользователя:**
   - base (64-69 ф): z real/chit med 3.82/2.84, потолок «0 потерь» t=2.73;
   - wide (637 ф): 5.06/3.84, потолок 3.58;
   - multi (18 языков, 20 ф): 3.29/1.78, потолок 2.35, t=2.73 → 33/360 потерь;
   - порог растёт с размером базы (~√(2·ln N)): глобальный фиксированный
     порог не переносится (t=2.73 режет multi, t=2.3 на wide убивает только 12/135).
   - `worth_asking` (счётчик слов) как admission убивает real: «what does HR24 do»
     → False. Формульные правила (is_greeting) — многоязычны, 0 потерь real.
   - z=None: в multi это 6/360 real + 38/90 chit (короткие сообщения).
   - **Открытые решения:** (а) z=None → fail-open?; (б) адаптивный t(N)=c·√(2·ln N)
     (c≈0.93) против фиксированного 2.3; (в) `admission: off` до holdout (блок 1).
4. **Блок 5 (латентность) готов** (`9a7a7a3`, `93c6f88`; `results/LATENCY_SWEEP.md`):
   p95 на 77.5% = реранк, `мс ≈ 17 + 10.6×docs` (r=0.994); **K=15 — дефолт**
   (wide p95 −45% при побайтово одинаковом качестве 72/86/4, перепрогон после
   фикса индекса); **jina-fp16 26/53 мс** против bge-q4 39/80 при recall 357/360 —
   рекомендация смены модели, второй llama-server на `:1236` НЕ запущен.
5. **Блок 6 (тесты) готов** (`64879fc`; `tests/test_holographic_rerank_gaps.py`,
   18 кейсов): HTTPError/URLError/JSONDecodeError, секрет не уходит на эндпоинт
   (prefetch-ветки), прямые тесты `_unified_stream`, мёртвый эндпоинт, файл-не-БД.
   Продакшн-багов не найдено.
6. **Ревью PR #10 получено (10 пунктов), гейт: «1-5 исправлены + замеры».**
   Проверено по коду — подтверждены как реальные баги/дыры:
   - **#1**: `_search` (`__init__.py:2213`) шлёт кандидаты в `order()` без
     `_sendable` (секреты видны search) → фикс + тест `OPENAI_API_KEY` не в payload;
   - **#2**: `rerank.py:216-222` — сокет-таймаут воркера (`TimeoutError`, не alive)
     не вызывает `_fail_locked` → нет паузы, `failures` не растёт;
   - **#4**: фолбэк `(lexical+meaning+entity)[:5]` ≠ старому пути; `_search:2214-2215`
     пересборка только из ≤40 индексов `order()` → хвост `limit` 41-100 теряется;
   - **#5**: `rerank: on` без `rerank_mode` → `settings_from` None (дефолт "off"),
     а UI показывает `reorder_before_gate`;
   - **#3** RECALL_MARGIN: off-path замер (emb=on, rerank off, 1.0 vs 0.6):
     hit 74→78, шум 79→80, wrong5 29=29 → **опубликовано в PR**
     (issuecomment-6343091582), решение о сплите — за ревьюером;
   - **#6** = наш «блок 4» (убить `override`, переименовать режимы, выкинуть
     инертный `rerank_margin`, чинить описание/README);
   - **#7** — частично закрыт блоком 6; осталось: identical-to-main (emb on/off),
     reorder реально меняет порядок, blend-порядок, deadline-тест (0.2→0.5 clamp);
   - **#8** — перегнать таблицу 433→432/30→6 на точном head + сырые JSONL
     (= **блок 1**), matched top-5 + счётчик выводов, фикс `recall@5` в
     `benchmark_rerank` (считает все строки), абляция окна δ, `rerank_window`
     в настройки, CPU: p95+модель+повторы;
   - **#9** — публичный сет ~200 фактов en/ru/zh (наш wide 616-637 подходит,
     оформить) + второй реранкер (jina, замер готов для GPU);
   - **#10** — housekeeping: версии не трогать, CHANGELOG → "Unreleased",
     выкинуть одноразовые пробы (`_en_meaning_debug.py`, `_margin_probe.py`,
     `score_probe.py`, `loss_probe.py`, `margin_sweep.py`, `variant_run.py`),
     README-строка про `rerank_weight: 0` и trust.

## План дальше (порядок)

1. Фиксы ревью **#1, #2, #4, #5** (маленькие, с тестами) — гейт draft→ready.
2. **Блок 1**: holdout-сет, Wilson CI/McNemar, версирование JSONL (git_sha),
   генерация PROVENANCE из jsonl, перегон на голове PR (#8).
3. **Блок 2**: CI-гейт (per-set floors ru≥25/en≥24/zh≥22/multi≥355, 2 прогона →
   побайтово одинаковый jsonl).
4. **Блок 4** = пункт #6: убрать `override`/мёртвые ветки, семпоток, `_http.py`,
   дедуп потоков; после него — остальное из #7 и #10.
5. **Admission (этап-0)**: после решений (а)/(б)/(в) — отдельный PR; `admission: off`
   до подтверждения holdout-блоком.
6. Оформление #9: wide как публичный сет, jina-замер, matched top-5.

## Артефакты

- Сводные отчёты: `results/PROVENANCE.md`, `results/LATENCY_SWEEP.md`,
  `results/BENCHMARK.md` (закоммичены).
- Сырые jsonl (закоммичены): `results/admission_{base,wide,multi}.jsonl`,
  `results/margin_offpath.jsonl`, `results/_lat_profile*.jsonl`,
  `results/_lat_{ksweep,widelat}*.json*`, `results/provenance_*.jsonl`.
- Воспроизведение off-path замера: `tools/_margin_offpath.py`.
- Старые `results/sweep*/`, `results/smoke/` — НЕ в git (устаревшие прогоны до
  фиксов; перегон по #8 заменит их).
- Статус ревьюера: ждём ответа на комментарий issuecomment-6343091582;
  он держит PR draft до исправления 1-5 и замеров.