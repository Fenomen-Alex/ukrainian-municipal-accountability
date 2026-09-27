# Multi-topic prevalence analysis

Analysis of the raw complaint corpus (`ml/data/{train,validation,test}.jsonl`) to decide how many
complaints are genuinely multi-topic and how to build a targeted multi-topic training set for the
v2 fine-tune. All numbers are produced by deterministic scripts over the corpus; nothing is
estimated or invented. Regenerate with the commands in the appendix.

## 1. The problem, restated in one number

`ml/tune/build_dataset.py` `to_json()` emits **exactly one topic per example**. As a result every
training, validation and test example in v1 has `len(topics) == 1`:

| split      | examples | topic count distribution |
|------------|---------:|--------------------------|
| train      |    5,384 | `{1: 5384}`              |
| validation |    1,124 | `{1: 1124}`              |
| test       |      329 | `{1: 329}`               |

The base model and the v1 adapter therefore never saw a multi-topic extraction target. The smoke
suite confirms the consequence: 0/4 multi-topic smoke cases passed for both the fine-tuned and the
base model (`ml/reports/smoke_test_report.md`).

## 2. Source-grounded signal: registration under multiple kinds

The raw corpus is curated from the municipality portal, where each record carries a single
`kind` (complaint category). Identical complaint text is sometimes registered under **two or three
different kinds** (same content, different `kind` rows) — the portal treating one complaint as
belonging to several departments. This is the only *source-grounded* evidence of multi-topic intent
we have without inventing labels.

Distinct normalized contents registered under ≥ N distinct kinds (across all three splits):

| min. distinct kinds | distinct contents |
|--------------------:|------------------:|
| ≥ 1                 | 6,535             |
| ≥ 2                 | 263               |
| ≥ 3                 | 16                |

- 247 contents have exactly 2 kinds, 16 have exactly 3.
- They span **41 distinct kind pairs/triples**. The dominant combinations (by number of texts):

| kind combination | texts |
|---|---|
| roads + sanitation              | 72 |
| water + sanitation              | 27 |
| housing + sanitation            | 23 |
| roads + electricity             | 15 |
| electricity + sanitation        | 12 |
| roads + water                   | 10 |
| water + housing                 | 10 |
| housing + electricity           | 9  |
| roads + housing                 | 8  |
| transport + sanitation          | 8  |

Rows that belong to a multi-kind text, by split:

| split      | rows in multi-kind texts |
|------------|-------------------------:|
| train      | 424                      |
| validation | 90                       |
| test       | 29                       |

Train-only multi-kind texts (not appearing in validation or test): **205**.

### 2.1 Noise in the signal

Multi-kind registration is *not* a clean label. Manual inspection of the 263 texts shows:

- **Genuine multi-topic** (text enumerates several independent problems, portal filed it under the
  matching departments) — e.g. `Б-967` (transport + park upkeep + riverside cleanliness), `П-2977`
  (pipe burst + courtyard road repair), `В-2702` (roof repair + facade funding + locked staircase),
  `Р-2312` (mowing + pot-hole patching).
- **Coding noise** (a single-problem text registered under an unrelated second kind) — e.g.
  `перерахунок за лютий за теплопостачання` registered under *roads*; `відновити електроопостачання
  у коридорах` registered under *payments*. These carry `kind` atoms that do not match the text.

So a text registered under two kinds is a *candidate*; the two kinds are a *domain hint*, not a
per-topic segmentation.

## 3. Explicit in-text multi-issue structure

A second, independent signal is the complaint text itself enumerating multiple issues:

- 24 multi-kind texts contain explicit structure markers such as `з двох питань`, `з кількох
  питань`, `звернувся з двох питань, а саме:`, `по-перше / по-друге`. These are unambiguously
  multi-topic.
- Plain `N)` enumeration markers are **not reliable** in this corpus: a regex for `1)` matches
  house/street numbers (`буд. №11`, `вул. № 3/4`) far more often than list items. Only 16 train
  records have two `N)` markers after inspection, most of which are number ranges, not question
  lists. `(а) ... (б) ...` letter enumerations were not found at all.
- **338** train texts mention ≥ 2 distinct street names; only **45** of those are also multi-kind
  registered. Multi-street is therefore a weak, mostly single-topic signal (one problem repeated
  across addresses — e.g. "відремонтувати дорогу по вул. X та вул. Y").

## 4. Corpus shape and duplicates

| split      | rows | distinct contents | rows that are dup content | dup uids (>1 row) |
|------------|-----:|------------------:|--------------------------:|------------------:|
| train      | 5,384 | 5,143            | 241                       | 322               |
| validation | 1,124 | 1,078            |  46                       |  44               |
| test       |   329 |   314            |  15                       |  14               |

All duplicates come from identical `(content, kind)` — a single complaint text registered several
times under the *same* kind — or identical content under *different* kinds (the multi-kind signal of
§2). Identical content across splits: **0** (train↔validation, train↔test, validation↔test).

Content length (train): median 296 chars, p90 473, max 1242 — long enough to contain several
independent problems.

## 5. Kind distribution in the training split

| kind                                       | rows |
|---                                        |-----:|
| Санітарний стан, благоустрій…  (sanitation) | 1695 |
| Гаряче та холодне водопостачання  (water)   |  785 |
| Будівництво та ремонт доріг, вулиць (roads) |  777 |
| Теплопостачання (heating)                   |  657 |
| Експлуатація та ремонт житла (housing)      |  610 |
| Електропостачання (electricity)             |  266 |
| Робота пасажирського транспорту (transport) |  259 |
| Плата за житло та комунальні послуги (payments) | 138 |
| Будівництво, містобудування (construction)  |   96 |
| Пільгове перевезення пасажирів (transport)  |   45 |
| Діяльність органів місцевого самоврядування (government) | 30 |
| Питання пов’язані з торгівлею (commerce)    |   26 |

No `benefits` records exist in any split.

## 6. Conclusions for the v2 dataset design

1. **True multi-topic prevalence is small but real**: roughly 1–4% of distinct texts show a
   genuine second independent problem (multi-kind candidate pool 263/6,535 = 4.0%; the subset that
   survives manual noise filtering is smaller). This matches the smoke failure being structural
   (never-trained behavior) rather than low data volume.
2. **Real examples take priority.** Build real multi-topic examples from the multi-kind pool,
   restricted to train-only content, manually splitting each text at its actual problem boundaries
   and verifying the kind hint against the segment. Every fact must trace to the source text.
3. **Synthetic mixtures fill the gap.** For attitudes where the corpus has too few clean real
   multi-topic examples, additionally construct mixtures by concatenating two *verified
   single-topic* train complaints that touch different topics, target = the union of their weak
   labels. Each mixture records both source `uid`s; nothing in a mixture is invented, but the
   *combination* is synthetic and must be flagged as such in provenance.
4. **Leakage control.** Multipurpose training examples must use train-only text (0 overlap with
   validation/test content, verified above). A held-out multi-topic suite for evaluation is built
   from *other* train-held-back multi-topic texts (or explicit mixtures) so the frozen v1
   validation/test benchmark stays untouched and comparative.
5. **Quantity target.** Decide the number of augmented examples from the observed prevalence, then
   keep the majority of training single-topic so the model does not start over-emitting topics on
   single-topic input. Target roughly 5–10% multi-topic in the augmented training set; the exact
   figure is fixed in `ml/data/tune/multitopic/README.md` and `meta.json`.

## Appendix: reproduction commands

```bash
.venv/bin/python - <<'EOF'
import json, re, collections
def norm(t): return re.sub(r"\s+"," ",re.sub(r"[^\w\s]"," ",t.lower())).strip()
# topic-count distribution over built v1 chat data:
for s in ("train","validation","test"):
    n=0; dist=collections.Counter()
    for l in open(f"ml/data/tune/{s}.jsonl"):
        ex=json.loads(l); c=len(ex["messages"][-1]["content"]); import json as j
        toks=j.loads(ex["messages"][-1]["content"])
        n+=1; dist[len(toks.get("topics",[]))]+=1
    print(s, n, dict(dist))
# multi-kind registration across splits:
tk=collections.defaultdict(set)
for s in ("train","validation","test"):
    for l in open(f"ml/data/{s}.jsonl"):
        tk[norm(json.loads(l).get("content") or "")].add(json.loads(l).get("kind"))
mk={t:ks for t,ks in tk.items() if len(ks)>1}
print("multi-kind contents:", len(mk), "3-kind:", sum(1 for ks in mk.values() if len(ks)>=3))
EOF
```