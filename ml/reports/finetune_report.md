# Демонстраційний експеримент: файнтюн Qwen3-8B 4-bit (MLX) для структурування звернень громадян

> **SUPERSEDED — historical v1 training report.**
> Canonical model: v2. See [`ml/tune/FINAL_STATUS.md`](../tune/FINAL_STATUS.md)
> and [`ml/tune/MODEL_EXPERIMENTS.md`](../tune/MODEL_EXPERIMENTS.md).
> **Do not re-run the training commands below.** Model development is frozen, and
> these commands target superseded corpora, hyperparameters and code paths.
> Numbers here are historical and describe v1.

**Дата:** 2026-09-24
**Середовище:** MacBook Pro (M1 Pro, 32 GB unified memory), macOS
**Стек:** MLX-LM 0.31.3, MLX 0.32.2, mlx-metal 0.32.2, Python 3.12.14 (`.venv-mlx`)
**Статус:** експеримент завершено повністю (дизайн → дані → тренування → оцінка → злитий артефакт)

---

## 1. Задача

З перетворенням довільних текстів звернень громадян українською (звернення до органів місцевого самоврядування) у структурований JSON:

```json
{
  "topics": [{
    "domain": "roads",
    "issue": "здійснення ремонту ямкового покриття по вул. Адама Міцкевича",
    "object": "вул. Адама Міцкевича",
    "requested_action": "",
    "attributes": {"street": "вул. Адама Міцкевича"}
  }]
}
```

## 2. Дані

- Джерело: `ml/data/{train,validation,test}.jsonl` (заморожені вхідні дані, не модифіковані).
- 6 837 унікальних рядків: 5 384 train / 1 124 valid / 329 test.
- **Слабкі мітки (`kind`)** від літералів категорій. `kind` → `domain` (карта `KIND_TO_DOMAIN`, 12 категорій, fallback `other`).
- **Принцип «не вчити галюцинувати»:** метадані (адреса/організація/статус/дата) майже не зустрічаються в тексті звернення (вулиця у тексті — 28% рядків, організація — 0%), тому `object` і `attributes` виводяться **виключно з тексту** (багатослівний regex вулиці), а не з колонок метаданих. Це навчає модель не вигадувати факти.
- `issue` — очищений текст із застосуванням PII-редагування (захищено спани вулиць, обрізано персональні дані). Перевірено: 0 PII-зіткнень у test.
- `requested_action` — речення з дієсловами-проханнями, інакше `""`.
- Усі 6 837 таргетів пройшли валідацію JSON-schema (Draft7, `annotation_schema.json`).
- Чат-формат (ChatDataset): `{"messages": [system, user, assistant], "uid"}`; літерали `kind` у системному промпті — це спостережувані в train значення, не дані самих звернень.

## 3. Вибір базової моделі

**Обрано: `mlx-community/Qwen3-8B-4bit`** (Apache-2.0, prefab 4-bit MLX, 133k завантажень).

| Модель | Ліцензія | Придатність | Рішення |
|---|---|---|---|
| Qwen3-8B 4-bit (MLX) | Apache-2.0 | відмінний JSON-дисциплін, гарна українська, готовий 4-bit квант | **обрано** |
| MamayLM-12B IT v2.0 | (Gemma TOU) | немає prefab 12B MLX-кванту (лише 27B), архітектура Gemma3 vision-tower потребує хірургії | відкладено як кандидат на 2-й етап |
| Mistral-7B 4-bit | Apache-2.0 | слабший Ukrainian + JSON, гірша стабільність | — |

Ноут: `Qwen/Qwen3-8B-Instruct-2507` **не існує**; правильний repo — `Qwen/Qwen3-8B`.

## 4. Тренування (QLoRA)

Команда (демонстраційний прогон, ~2 години):

```
.venv-mlx/bin/python -m ml.tune.run_train \
  --model mlx-community/Qwen3-8B-4bit \
  --data ml/data/tune \
  --adapter-path ml/data/tune/adapters/qwen3-8b-lora \
  --iters 400 --batch-size 2 --grad-checkpoint \
  --grad-accumulation-steps 2 --learning-rate 1e-4 \
  --fine-tune-type lora --num-layers 16
```

- `ml/tune/run_train.py` — обгортка над `mlx_lm.lora`; **критично**: токенізатор обгорнуто `_NoThinkingTokenizer`, що форсує `enable_thinking=False` у шаблоні (Qwen3 інакше вставляє ` thinking\n` у таргет; без цього train-формат розходився б з inference).
- Треновані параметри: 9.699M / 8190.735M (0.118%), LoRA rank=8, scale=20, 16 шарів, AdamW lr=1e-4, `mask_prompt=True` (loss лише по assistant-таргету), batch 2 × grad-accum 2.
- Дані: `ml/data/tune/{train,valid,test}.jsonl` (valid — симлінк на validation).
- Як проходив train loss: 0.518 (iter 10) → 0.034 (iter 400); val loss: 1.190 (iter 1) → 0.041 (iter 400). Плато ~0.04 без явного переважування.
- Швидкість: ~17-19 токен/с (em по masked-токенам), It/sec ~0.05, Peak mem 8.8 GB. 400 ітерацій ≈ 2 год.

**Артефакти:**
- `ml/data/tune/adapters/qwen3-8b-lora/adapters.safetensors` (≈39 MB) + `adapter_config.json`, чекпоінти 00001xx/…/0000400
- **Злита модель:** `ml/data/tune/adapters/qwen3-8b-lora-fused/` (≈4.6 GB, 4-bit) — готова для inference без окремого адаптера.

## 5. Оцінка (`ml/data/tune/eval/`)

329 тестових записів; усі системи оцінено позиційно (test містить 14 дубльованих UID).

| метрика | deterministic* | base (без файнтюну) | **lora (файнтюн)** |
|---|---|---|---|
| json_parse_rate | 1.000 | 0.988 | **0.997** |
| schema_validity_rate | 1.000 | 0.973 | **0.991** |
| domain_accuracy | 1.000 | 0.778 | **0.845** |
| domain_macro_f1 | 1.000 | 0.495 | **0.645** |
| issue_rouge_l | 1.000 | 0.124 | **0.901** |
| object_exact | 1.000 | 0.067 | **0.921** |
| object_token_overlap | 1.000 | 0.281 | **0.944** |
| action_presence_match | 1.000 | 0.587 | **0.973** |
| hallucination_rate | 0.000 | 0.739 | **0.024** |

\* deterministic = той самий weak-label трансформатор, який породив таргети; це верхня межа (1.0) для цього набору і «чесний базового» рівень.

**Висновок:** файнтюн перетворив модель з «майже нечитабельного» на робочу. Найголовніше — різке зниження галюцинацій (73.9% → 2.4%) і майже ідеальний витяг об'єкта/атрибутів (object_exact 6.7% → 92.1%). JSON-структура стабільна (parse 99.7%, schema 99.1%). Відкриті слабкості: domain_accuracy 84.5% (макро-F1 0.65) — частина помилок у межових категоріях (transport/sanitation), і `requested_action` присутній лише у ~50% таргетів (метрика чесна до цього).

## 6. Приклад inference

```
$ mlx generate --model ml/data/tune/adapters/qwen3-8b-lora-fused
SRC: «Щодо здійснення ремонту ямкового покриття по вул. Адама Міцкевича. Відповідь заявнику...»
OUT (8.1s, temp=0):
{"topics": [{"domain": "roads", "issue": "здійснення ремонту ямкового покриття по вул. Адама Міцкевича",
             "object": "вул. Адама Міцкевича", "requested_action": "",
             "attributes": {"street": "вул. Адама Міцкевича"}}]}
```

(злита модель запускається напряму через `mlx_lm.generate`/`--model` на шлях `-fused`, без `--adapter-path`.)

## 7. Тести

`ml/tests/test_tune_dataset.py` — 16 тестів (валідність схеми, чесність текст-обмежених міток, PII-редагування, верхня межа deterministic, позиційна оцінка при дублікатах UID, `_NoThinkingTokenizer`). Повний прогон: **166 passed + 4 subtests** за 16s.

## 8. Обмеження та наступні кроки

- Слабкі мітки = слабкі, тому absolute quality обмежена якістю таргетів (0.845 domain_acc — стеля цього датасету не 1.0). Потрібна ручна золота вибірка для чесної оцінки.
- 400 ітерацій — демонстраційний обсяг; більше ітерацій / більше даних піднімуть domain_acc.
- MamayLM-12B — кандидат на 2-й етап з точки зору ліцензії/якості української (потребує MLX-конвертації 12B і обробки Gemma-адаптації).
- Інференс: ~8-12s на запис/600 токенів → для продакшену варто подивитися на 4-bit, streaming, batch, і можливий 3/7B-тредофф.

## 9. Відтворюваність

```
.venv-mlx/bin/python -m ml.tune.run_train --adapter-path ml/data/tune/adapters/qwen3-8b-lora --iters 400
.venv-mlx/bin/python -m ml.tune.run_eval --mode base
.venv-mlx/bin/python -m ml.tune.run_eval --mode lora --adapter-path ml/data/tune/adapters/qwen3-8b-lora
.venv-mlx/bin/mlx_lm fuse --model mlx-community/Qwen3-8B-4bit \
  --adapter-path ml/data/tune/adapters/qwen3-8b-lora \
  --save-path ml/data/tune/adapters/qwen3-8b-lora-fused
```