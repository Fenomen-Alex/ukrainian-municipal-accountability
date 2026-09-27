# Smoke-test report: fine-tuned vs base model (targeted suite)

**Scope:** 20 hand-written Ukrainian municipal complaints, run at temperature 0.0, `max_tokens=800`, same system prompt and Qwen3 chat template (`enable_thinking=False`) as training.

**Warning:** this is a qualitative / targeted smoke suite (N=20, hand-authored, no platform gold labels). It is NOT a statistically rigorous benchmark. It is meant to surface behavioural differences and regressions for human inspection.

The smoke cases use a *richer* free-form domain vocabulary (`lighting`, `water_leak`, ...) than the model's 13 schema domains. Comparison maps each expected domain to acceptable schema domains via `EXPECTED_TO_SCHEMA` in `ml/tune/eval_smoke.py`. A case PASSes if schema-valid AND full expected-domain coverage AND no hallucination flags AND (for multi-topic cases) >=2 distinct topics.

## Aggregate

| system | n | passed | schema-valid | full coverage | with hallucination | multi-topic (exp) | multi-topic (ok) |
|---|---|---|---|---|---|---|---|
| finetuned | 20 | 17 | 20 | 18 | 0 | 4 | 1 |
| base | 20 | 8 | 20 | 17 | 10 | 4 | 0 |

## Failure patterns

### finetuned
- **Multi-topic collapse:** 3/4 multi-topic cases emitted only one topic (smoke-14, smoke-15, smoke-16).
- **Hallucination flags:** 0 cases (none).

### base
- **Multi-topic collapse:** 4/4 multi-topic cases emitted only one topic (smoke-13, smoke-14, smoke-15, smoke-16).
- **Hallucination flags:** 10 cases (smoke-03, smoke-06, smoke-07, smoke-08, smoke-09, smoke-10, smoke-12, smoke-15, smoke-16, smoke-18).
  - smoke-03: topic[0].attributes:неперевірена
  - smoke-06: topic[0].attributes:ст. 125; topic[0].attributes:2023-04-05
  - smoke-07: topic[0].object; topic[0].requested_action; topic[0].attributes:школа №5
  - smoke-08: topic[0].requested_action; topic[0].attributes:виявлено
  - smoke-09: topic[0].attributes:відкрите
  - smoke-10: topic[0].attributes:ніч; topic[0].attributes:впавти
  - smoke-12: topic[0].attributes:Осьовий період
  - smoke-15: topic[0].attributes:прорив
  - smoke-16: topic[0].attributes:неперевідена дата
  - smoke-18: topic[0].attributes:міська рада


## Per-case detail (fine-tuned vs base)

| id | category | expected | system | domains | object | action | attributes | flags |
|---|---|---|---|---|---|---|---|---|
| smoke-01 | Standard Concrete | roads | finetuned | roads | вулиці Шевченка | ∅ | {"street": "вулиці Шевченка"} | – |
| smoke-01 | Standard Concrete | roads | base | roads | вул. Шевченка, будинок №24 | провести ямковий ремонт асфаль | {"вулиця": "Шевченка", "будино | – |
| smoke-02 | Standard Concrete | lighting | finetuned | electricity | вул. Героїв Майдану | ∅ | {"street": "вул. Героїв Майдан | – |
| smoke-02 | Standard Concrete | lighting | base | electricity | вул. Героїв Майдану 12-16 | Відновлення роботи ліхтарів | {"address": "вул. Героїв Майда | – |
| smoke-03 | Standard Concrete | water_supply | finetuned | water | проспект Свободи | ∅ | {"street": "проспект Свободи"} | – |
| smoke-03 | Standard Concrete | water_supply | base | water | проспект Свободи 45, під'їзд №2 | ліквідувати порив водопроводу | {"статус": "вже другий день",  | H:topic[0].attributes:неперевірена |
| smoke-04 | Standard Concrete | waste_management | finetuned | sanitation | вул. Соборній | ∅ | {"street": "вул. Соборній"} | – |
| smoke-04 | Standard Concrete | waste_management | base | sanitation | вул. Соборній 88 | Забезпечити вивіз побутових ві | {"status": "переповнені", "dat | – |
| smoke-05 | Standard Concrete | landscaping | finetuned | sanitation | вул. Яновського навпроти Дендропарку суха гіл | ∅ | {"street": "вул. Яновського на | – |
| smoke-05 | Standard Concrete | landscaping | base | sanitation | вул. Яновського навпроти Дендропарку | Виконати кронування або аварій | {"object": "вул. Яновського на | – |
| smoke-06 | Short & Terse | hot_water | finetuned | water | ∅ | ∅ | {} | – |
| smoke-06 | Short & Terse | hot_water | base | water | Комарова 14 кв 5 | ∅ | {"статья": "ст. 125", "дата":  | H:topic[0].attributes:ст. 125;topic[0].attributes:2023-04-05 |
| smoke-07 | Short & Terse | waste_management | finetuned | sanitation | ∅ | ∅ | {} | – |
| smoke-07 | Short & Terse | waste_management | base | sanitation | школа №5 | перемістити смітник | {"object": "школа №5"} | H:topic[0].object;topic[0].requested_action;topic[0].attributes:школа №5 |
| smoke-08 | Short & Terse | roads | finetuned | roads | вул. Соборній | ∅ | {"street": "вул. Соборній"} | – |
| smoke-08 | Short & Terse | roads | base | roads | вул. Соборній, зупинка 'Пошта' | виконати ремонт дороги | {"статус": "виявлено"} | H:topic[0].requested_action;topic[0].attributes:виявлено |
| smoke-09 | Messy & Emotional | road_work_pedestrian | finetuned | sanitation | вул. Перемоги і постійно чіпляюся за розриту  | ∅ | {"street": "вул. Перемоги і по | – |
| smoke-09 | Messy & Emotional | road_work_pedestrian | base | sanitation | вул. Перемоги, будинок 10 | закопати траншею та зробити но | {"статус": "відкрите", "дата": | H:topic[0].attributes:відкрите |
| smoke-10 | Messy & Surzhyk | manholes_infrastructure | finetuned | sanitation | ∅ | ∅ | {} | – |
| smoke-10 | Messy & Surzhyk | manholes_infrastructure | base | sanitation | вул. Полтавська, район Ковальовка | закрити люк | {"сторона": "ніч", "ризик": "в | H:topic[0].attributes:ніч;topic[0].attributes:впавти |
| smoke-11 | Messy & Sarcastic | water_leak | finetuned | water | вул. Гагаріна | ∅ | {"street": "вул. Гагаріна"} | – |
| smoke-11 | Messy & Sarcastic | water_leak | base | water | вул. Гагаріна 5 | Відремонтувати трубу | {"comment": "Басейн у дворі на | – |
| smoke-12 | Messy & Rambling | playgrounds | finetuned | sanitation | вул. Академіка Корольова | ∅ | {"street": "вул. Академіка Кор | – |
| smoke-12 | Messy & Rambling | playgrounds | base | sanitation | вул. Академіка Корольова 3 | Відремонтувати або демонтувати | {"status": "Аварійні елементи" | H:topic[0].attributes:Осьовий період |
| smoke-13 | Multi-Topic | lighting,roads | finetuned | electricity,roads | ∅ | ∅ | {} | – |
| smoke-13 | Multi-Topic | lighting,roads | base | roads | Вулиця Незалежності від будинку 1 до 15 | Відновити освітлення та відрем | {"object": "Вулиця Незалежност | uncovered:lighting, multi-fail |
| smoke-14 | Multi-Topic | waste_management,animals | finetuned | sanitation | вул. Паркова | ∅ | {"street": "вул. Паркова"} | uncovered:animals, multi-fail |
| smoke-14 | Multi-Topic | waste_management,animals | base | sanitation | вул. Паркова 12 | Вивезти сміття та викликати сл | {"issue": "переповнені контейн | uncovered:animals, multi-fail |
| smoke-15 | Multi-Topic | elevators,sewage_basement | finetuned | housing | Вул. Київська | ∅ | {"street": "Вул. Київська"} | multi-fail |
| smoke-15 | Multi-Topic | elevators,sewage_basement | base | housing | Вул. Київська 40, під'їзд 1 | Відремонтувати ліфт та відкача | {"стать": "підвал", "статус":  | H:topic[0].attributes:прорив, multi-fail |
| smoke-16 | Multi-Topic | landscaping,traffic_signs | finetuned | roads | вул. Гоголя та Лесі Українки гілки дерев повн | ∅ | {"street": "вул. Гоголя та Лес | uncovered:landscaping, multi-fail |
| smoke-16 | Multi-Topic | landscaping,traffic_signs | base | roads | перехрестя вул. Гоголя та Лесі Українки | обрізати гілки та вирівняти зн | {"status": "похилився", "date" | uncovered:landscaping, H:topic[0].attributes:неперевідена дата, multi-fail |
| smoke-17 | Implicit Action | hot_water_leak | finetuned | water | вул. Яновського | ∅ | {"street": "вул. Яновського"} | – |
| smoke-17 | Implicit Action | hot_water_leak | base | water | Дендропарк на вул. Яновського | ∅ | {"street": "Яновського", "obje | – |
| smoke-18 | Vague Location | traffic_lights | finetuned | roads | ∅ | ∅ | {} | – |
| smoke-18 | Vague Location | traffic_lights | base | transport | зупинка у центрі міста | ∅ | {"організація": "міська рада"} | H:topic[0].attributes:міська рада |
| smoke-19 | Non-Municipal / Private | housing_dispute_or_other | finetuned | housing | вул. Миру | ∅ | {"street": "вул. Миру"} | – |
| smoke-19 | Non-Municipal / Private | housing_dispute_or_other | base | housing | вул. Миру 5 | зробити з сусідом щось | {"квартира": "12", "об’єкт": " | – |
| smoke-20 | Transport & Schedule | public_transport | finetuned | transport | ∅ | ∅ | {} | – |
| smoke-20 | Transport & Schedule | public_transport | base | transport | зупинка 'Автовокзал' | збільшити кількість машин на м | {"маршрут": "114", "зупинка":  | – |

## Qualitative notes

### finetuned

**smoke-01** [Standard Concrete] expected=roads
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "roads", "issue": "вулиці Шевченка, біля будинку №24, утворилася глибока яма на проїжджій частині. провести ямковий ремонт асфальтного покриття.", "object": "вулиці Шевченка", "requested_action": "", "attributes": {"street": "вулиці Шевченка"}}]}`

**smoke-02** [Standard Concrete] expected=lighting
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "electricity", "issue": "Вже тиждень не працює вуличне освітлення за адресою вул. Героїв Майдану 12-16. будь ласка, роботу ліхтарів.", "object": "вул. Героїв Майдану", "requested_action": "", "attributes": {"street": "вул. Героїв Майдану"}}]}`

**smoke-03** [Standard Concrete] expected=water_supply
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "water", "issue": "проспект Свободи 45 витікає холодна вода з люка перед під'їздом №2. тече вже другий день. ліквідувати порив водопроводу.", "object": "проспект Свободи", "requested_action": "", "attributes": {"street": "проспект Свободи"}}]}`

**smoke-04** [Standard Concrete] expected=waste_management
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "sanitation", "issue": "будинку по вул. Соборній 88 переповнені сміттєві контейнери, сміття не вивозять з минулої п'ятниці. забезпечити вивіз побутових відходів.", "object": "вул. Соборній", "requested_action": "", "attributes": {"street": "вул. Соборній"}}]}`

**smoke-05** [Standard Concrete] expected=landscaping
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "sanitation", "issue": "вул. Яновського навпроти Дендропарку суха гілка дерева загрожує падінням на пішохідний тротуар. виконати кронування або аварійне обрізання гілки.", "object": "вул. Яновського навпроти Дендропарку суха гілка дерева загрожує падінням на пішохідний тротуар", "requested_action": "", "attributes": {"street": "вул. Яновського навпроти Дендропарку суха гілка`

**smoke-06** [Short & Terse] expected=hot_water
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "water", "issue": "14 кв 5.", "object": "", "requested_action": "", "attributes": {}}]}`

**smoke-07** [Short & Terse] expected=waste_management
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "sanitation", "issue": "нарешті смітник біля школи №5!", "object": "", "requested_action": "", "attributes": {}}]}`

**smoke-08** [Short & Terse] expected=roads
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "roads", "issue": "яма біля зупинки 'Пошта' на вул. Соборній.", "object": "вул. Соборній", "requested_action": "", "attributes": {"street": "вул. Соборній"}}]}`

**smoke-09** [Messy & Emotional] expected=road_work_pedestrian
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "sanitation", "issue": "з дитиною у візочку біля будинку 10 по вул. Перемоги і постійно чіпляюся за розриту траншею, яку розкопали ще місяць тому і кинули! ледь не випала! закопайте цей жах і зробіть нормальний тротуар!", "object": "вул. Перемоги і постійно чіпляюся за розриту траншею", "requested_action": "", "attributes": {"street": "вул. Перемоги і постійно чіпляюся за ро`

**smoke-10** [Messy & Surzhyk] expected=manholes_infrastructure
- schema_valid=True notes=['surzhyk input']
- raw: `{"topics": [{"domain": "sanitation", "issue": "закрийте люк, бо туда хтось упаде ноччю, кришки нема вообще!", "object": "", "requested_action": "", "attributes": {}}]}`

**smoke-11** [Messy & Sarcastic] expected=water_leak
- schema_valid=True notes=['sarcasm input']
- raw: `{"topics": [{"domain": "water", "issue": "водоканалу за безкоштовний басейн у дворі на вул. Гагаріна 5. вже час відремонтувати трубу, бо скоро будуть плавати качки?", "object": "вул. Гагаріна", "requested_action": "", "attributes": {"street": "вул. Гагаріна"}}]}`

**smoke-12** [Messy & Rambling] expected=playgrounds
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "sanitation", "issue": "звертаюся до вас вже втретє. минулого року писали за дитячий майданчик на вул. Академіка Корольова 3, де зламані гойдалки. Обіцяли відремонтувати до літа, але вже осінь, а гойдалки так і висять на одній цепі. відремонтувати або демонтувати аварійні елементи.", "object": "вул. Академіка Корольова", "requested_action": "", "attributes": {"street": "вул.`

**smoke-13** [Multi-Topic] expected=lighting,roads
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "electricity", "issue": "вулиці Незалежності від будинку 1 до 15 не працює жоден вуличний ліхтар", "object": "", "requested_action": "", "attributes": {}}, {"domain": "roads", "issue": "на всій цій ділянці глибокі ями на дорозі", "object": "", "requested_action": "", "attributes": {}}]}`

**smoke-14** [Multi-Topic] expected=waste_management,animals
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "sanitation", "issue": "вивезти сміття та викликати службу вилову собак", "object": "вул. Паркова", "requested_action": "", "attributes": {"street": "вул. Паркова"}}]}`

**smoke-15** [Multi-Topic] expected=elevators,sewage_basement
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "housing", "issue": "40, під'їзд 1: зламався пасажирський ліфт (не їде вище 2 поверху), а в підвалі прорвало каналізацію і стоїть жахливий сморід. відремонтувати ліфт та відкачати воду з підвалу.", "object": "Вул. Київська", "requested_action": "", "attributes": {"street": "Вул. Київська"}}]}`

**smoke-16** [Multi-Topic] expected=landscaping,traffic_signs
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "roads", "issue": "перехрестя вул. Гоголя та Лесі Українки гілки дерев повністю закрили дорожній знак 'Уступи дорогу', а сам знак похилився після вітру. обрізати гілки та вирівняти знак.", "object": "вул. Гоголя та Лесі Українки гілки дерев повністю закрили дорожній знак", "requested_action": "", "attributes": {"street": "вул. Гоголя та Лесі Українки гілки дерев повністю зак`

**smoke-17** [Edge Case - Implicit Action] expected=hot_water_leak
- schema_valid=True notes=['implicit-action input']
- raw: `{"topics": [{"domain": "water", "issue": "прорвало трубу гарячого водопостачання біля Дендропарку на вул. Яновського. всю дорогу затопило окропом.", "object": "вул. Яновського", "requested_action": "", "attributes": {"street": "вул. Яновського"}}]}`

**smoke-18** [Edge Case - Vague Location] expected=traffic_lights
- schema_valid=True notes=['vague-location input']
- raw: `{"topics": [{"domain": "roads", "issue": "центрі міста біля зупинки не працює світлофор і утворюються затори.", "object": "", "requested_action": "", "attributes": {}}]}`

**smoke-19** [Edge Case - Non-Municipal / Private] expected=housing_dispute_or_other
- schema_valid=True notes=['out-of-scope/private input']
- raw: `{"topics": [{"domain": "housing", "issue": "зробіть з ним щось!", "object": "вул. Миру", "requested_action": "", "attributes": {"street": "вул. Миру"}}]}`

**smoke-20** [Edge Case - Transport & Schedule] expected=public_transport
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "transport", "issue": "№114 курсує з порушенням інтервалу: замість 10 хвилин чекаємо по 40 хвилин на зупинці 'Автовокзал', а водії грубіянять пасажирам. Збільшій кількість машин на маршруті №114 та проведіть бесіду з перевізником.", "object": "", "requested_action": "", "attributes": {}}]}`

### base

**smoke-01** [Standard Concrete] expected=roads
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "roads", "issue": "утворення глибокої ями на проїжджій частині", "object": "вул. Шевченка, будинок №24", "requested_action": "провести ямковий ремонт асфальтного покриття", "attributes": {"вулиця": "Шевченка", "будинок": "24"}}]}`

**smoke-02** [Standard Concrete] expected=lighting
- schema_valid=True notes=–
- raw: `{   "topics": [     {       "domain": "electricity",       "issue": "Вуличне освітлення не працює",       "object": "вул. Героїв Майдану 12-16",       "requested_action": "Відновлення роботи ліхтарів",       "attributes": {         "address": "вул. Героїв Майдану 12-16"       }     }   ] }`

**smoke-03** [Standard Concrete] expected=water_supply
- schema_valid=True notes=–
- raw: `{   "topics": [     {       "domain": "water",       "issue": "витікання холодної води з люка",       "object": "проспект Свободи 45, під'їзд №2",       "requested_action": "ліквідувати порив водопроводу",       "attributes": {         "статус": "вже другий день",         "дата": "неперевірена"       }     }   ] }`

**smoke-04** [Standard Concrete] expected=waste_management
- schema_valid=True notes=–
- raw: `{   "topics": [     {       "domain": "sanitation",       "issue": "Сміттєві контейнери переповнені",       "object": "вул. Соборній 88",       "requested_action": "Забезпечити вивіз побутових відходів",       "attributes": {         "status": "переповнені",         "date": "минулої п'ятниці"       }     }   ] }`

**smoke-05** [Standard Concrete] expected=landscaping
- schema_valid=True notes=–
- raw: `{   "topics": [     {       "domain": "sanitation",       "issue": "Суха гілка дерева загрожує падінням на пішохідний тротуар",       "object": "вул. Яновського навпроти Дендропарку",       "requested_action": "Виконати кронування або аварійне обрізання гілки",       "attributes": {         "object": "вул. Яновського навпроти Дендропарку"       }     }   ] }`

**smoke-06** [Short & Terse] expected=hot_water
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "water", "issue": "Немає гарячої води", "object": "Комарова 14 кв 5", "requested_action": "", "attributes": {"статья": "ст. 125", "дата": "2023-04-05"}}]}`

**smoke-07** [Short & Terse] expected=waste_management
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "sanitation", "issue": "неправильно розташований смітник", "object": "школа №5", "requested_action": "перемістити смітник", "attributes": {"object": "школа №5"}}]}`

**smoke-08** [Short & Terse] expected=roads
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "roads", "issue": "Яма на дорозі", "object": "вул. Соборній, зупинка 'Пошта'", "requested_action": "виконати ремонт дороги", "attributes": {"статус": "виявлено"}}]}`

**smoke-09** [Messy & Emotional] expected=road_work_pedestrian
- schema_valid=True notes=–
- raw: `{   "topics": [     {       "domain": "sanitation",       "issue": "розкопане отвідки траншею на тротуарі",       "object": "вул. Перемоги, будинок 10",       "requested_action": "закопати траншею та зробити нормальний тротуар",       "attributes": {         "статус": "відкрите",         "дата": "місяць тому"       }     }   ] }`

**smoke-10** [Messy & Surzhyk] expected=manholes_infrastructure
- schema_valid=True notes=['surzhyk input']
- raw: `{"topics": [{"domain": "sanitation", "issue": "відсутність криків на люках", "object": "вул. Полтавська, район Ковальовка", "requested_action": "закрити люк", "attributes": {"сторона": "ніч", "ризик": "впавти"}}]}`

**smoke-11** [Messy & Sarcastic] expected=water_leak
- schema_valid=True notes=['sarcasm input']
- raw: `{   "topics": [     {       "domain": "water",       "issue": "Тріска в системі водопостачання",       "object": "вул. Гагаріна 5",       "requested_action": "Відремонтувати трубу",       "attributes": {         "comment": "Басейн у дворі на вул. Гагаріна 5"       }     }   ] }`

**smoke-12** [Messy & Rambling] expected=playgrounds
- schema_valid=True notes=–
- raw: `{   "topics": [     {       "domain": "sanitation",       "issue": "Зламані гойдалки на дитячому майданчику",       "object": "вул. Академіка Корольова 3",       "requested_action": "Відремонтувати або демонтувати аварійні елементи",       "attributes": {         "status": "Аварійні елементи",         "date": "Осьовий період"       }     }   ] }`

**smoke-13** [Multi-Topic] expected=lighting,roads
- schema_valid=True notes=–
- raw: `{   "topics": [     {       "domain": "roads",       "issue": "Відсутнє освітлення та ями на дорозі",       "object": "Вулиця Незалежності від будинку 1 до 15",       "requested_action": "Відновити освітлення та відремонтувати дорогу",       "attributes": {         "object": "Вулиця Незалежності від будинку 1 до 15"       }     }   ] }`

**smoke-14** [Multi-Topic] expected=waste_management,animals
- schema_valid=True notes=–
- raw: `{   "topics": [     {       "domain": "sanitation",       "issue": "Переповнені контейнери для сміття у дворі призводять до появи безпритульних собак",       "object": "вул. Паркова 12",       "requested_action": "Вивезти сміття та викликати службу вилову собак",       "attributes": {         "issue": "переповнені контейнери для сміття",         "object": "вул. Паркова 12",         "requested_acti`

**smoke-15** [Multi-Topic] expected=elevators,sewage_basement
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "housing", "issue": "Не роботи пасажирський ліфт та каналізаційна прорива в підвалі", "object": "Вул. Київська 40, під'їзд 1", "requested_action": "Відремонтувати ліфт та відкачати воду з підвалу", "attributes": {"стать": "підвал", "статус": "прорив", "обладнання": "ліфт"}}]}`

**smoke-16** [Multi-Topic] expected=landscaping,traffic_signs
- schema_valid=True notes=–
- raw: `{   "topics": [     {       "domain": "roads",       "issue": "Дорожній знак 'Уступи дорогу' приховано гілками дерев та похилився",       "object": "перехрестя вул. Гоголя та Лесі Українки",       "requested_action": "обрізати гілки та вирівняти знак",       "attributes": {         "status": "похилився",         "date": "неперевідена дата"       }     }   ] }`

**smoke-17** [Edge Case - Implicit Action] expected=hot_water_leak
- schema_valid=True notes=['implicit-action input']
- raw: `{   "topics": [     {       "domain": "water",       "issue": "Прорив труби гарячого водопостачання",       "object": "Дендропарк на вул. Яновського",       "requested_action": "",       "attributes": {         "street": "Яновського",         "object": "Дендропарк"       }     }   ] }`

**smoke-18** [Edge Case - Vague Location] expected=traffic_lights
- schema_valid=True notes=['vague-location input']
- raw: `{"topics": [{"domain": "transport", "issue": "відсутність роботи світлофора призводить до заторів", "object": "зупинка у центрі міста", "requested_action": "", "attributes": {"організація": "міська рада"}}]}`

**smoke-19** [Edge Case - Non-Municipal / Private] expected=housing_dispute_or_other
- schema_valid=True notes=['out-of-scope/private input']
- raw: `{   "topics": [     {       "domain": "housing",       "issue": "шум з сусідньої квартири",       "object": "вул. Миру 5",       "requested_action": "зробити з сусідом щось",       "attributes": {         "квартира": "12",         "об’єкт": "вул. Миру 5"       }     }   ] }`

**smoke-20** [Edge Case - Transport & Schedule] expected=public_transport
- schema_valid=True notes=–
- raw: `{"topics": [{"domain": "transport", "issue": "маршрутка не відповідає інтервалу", "object": "зупинка 'Автовокзал'", "requested_action": "збільшити кількість машин на маршруті №114 та проведіть бесіду з перевізником", "attributes": {"маршрут": "114", "зупинка": "Автовокзал", "інтервал": "10 хвилин", "поточний інтервал": "40 хвилин"}}]}`

---
_Generated by `ml/tune/eval_smoke.py`. Raw outputs live in `ml/data/tune/smoke/results/{finetuned,base}/smoke_results.jsonl`._