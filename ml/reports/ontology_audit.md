# Ontology audit — 12 labels, 13 domains, 11 buckets in practice

The model cannot choose a domain the portal taxonomy does not offer, because the
training `domain` is a lookup of the portal `kind` rather than a reading of the
text. This report measures what that costs.

## 1. The mismatch, precisely

| set | size | contents |
| --- | --- | --- |
| `ml/data/labels.json` entries | 12 | portal `kind` strings |
| `KIND_TO_DOMAIN` entries | 12 | 12 kinds → domains |
| distinct domains in `KIND_TO_DOMAIN` | **11** | `transport` is the target of two kinds |
| schema enum | **13** | `roads, water, heating, housing, transport, sanitation, electricity, construction, benefits, government, commerce, payments, other` |

Two consequences:

- **`benefits` and `other` are unreachable.** No kind maps to them, so they have zero
  training support and no defensible weak-label rule. They stay in the schema because
  removing them would invalidate every recorded metric.
- **`transport` absorbs two kinds** — `Пільгове перевезення пасажирів` (privileged
  passenger transport) and `Робота пасажирського транспорту` (public transit
  operations). Complaints about a subsidised bus pass and complaints about a
  timetable are the same training label.

## 2. The `sanitation` bucket is a dumping ground

Because the label is a function of `kind`, the model's domain distribution is the
portal's, and one portal kind absorbs a large share of the corpus. Keyword probe over
all 6837 rows, showing the resulting domain distribution:

| keyword probe | rows | domain distribution |
| --- | --- | --- |
| tree / bush / branch / mowing | 882 | **sanitation 88 %**, roads 4 %, transport 3 % |
| waste / dumpster | 588 | **sanitation 87 %**, roads 5 %, payments 4 % |
| playground | 227 | **sanitation 85 %**, roads 8 %, construction 2 % |
| stray animal | 48 | **sanitation 92 %**, transport 4 %, roads 2 % |
| bus / trolleybus / stop | 394 | transport 64 %, **sanitation 27 %**, roads 7 % |
| traffic light | 22 | **sanitation 64 %**, roads 27 %, electricity 5 % |
| street lighting | 234 | electricity 79 %, **sanitation 8 %**, roads 7 % |
| sidewalk | 261 | **sanitation 59 %**, roads 31 %, construction 4 % |
| pothole / asphalt | 543 | roads 75 %, **sanitation 16 %**, water 3 % |
| lift | 325 | housing 91 %, electricity 3 %, payments 2 % |
| leak / burst pipe | 440 | water 61 %, **housing 17 %**, sanitation 11 % |
| heating | 844 | heating 81 %, sanitation 5 %, payments 5 % |

The pattern is consistent: `sanitation` is the residual bucket for anything the
portal files under `Санітарний стан, благоустрій населених пунктів, прибудинкових
територій`, which in practice covers fallen trees, dumped waste, broken play
equipment, stray animals, defective traffic lights and a slice of sidewalks.

**Traffic lights are the sharpest case.** A dead traffic light is an
electricity/traffic-safety problem. The portal files it under sanitation, so the
model is trained to answer `sanitation`. The one traffic-light case in the frozen test
set is predicted `electricity` — the model is right and the label is wrong. This is
the clearest single demonstration that a ceiling on domain accuracy is being imposed
by the ontology rather than by the model.

## 3. Measured cost on the frozen test set

First-topic domain, expected (from `kind`) vs predicted, over the 314 predictions that
parsed to at least one topic:

| expected → predicted | count |
| --- | --- |
| sanitation → roads | 15 |
| sanitation → water | 9 |
| sanitation → housing | 5 |
| housing → sanitation | 3 |
| roads → sanitation | 3 |
| housing → water | 3 |
| construction → housing | 2 |
| other single confusions | 9 |

49 / 314 = 15.6 % first-topic domain mismatches, consistent with the recorded
`domain_accuracy` 0.8389. The confusion is overwhelmingly **`sanitation` against its
neighbours** — exactly the buckets §2 shows are semantically adjacent. The model is
not confusing unrelated things; it is guessing between labels that the ontology makes
nearly indistinguishable.

`domain_macro_f1` is 0.7596 against `domain_accuracy` 0.8389. The 8-point gap is the
signature of imbalanced buckets: accuracy is dominated by `sanitation` and the
large classes, while macro-F1 weights every domain equally, including the ones with
little support.

## 4. Multi-topic interaction

On the 85-case multitopic suite the same ontology pressure shows up as domain
substitution rather than count error. Of the 85:

- 58 (68.2 %) get both the count and the domain set exactly right
- 12 (14.1 %) get the topic count wrong — 8 under-extractions (predicted 1 of 2),
  3 over-extractions (3 or 4), 1 empty
- 16 (18.8 %) get the count right but substitute a domain
- 3 predict two topics that share one domain

The 16 substitutions are dominated by pairs that §2 identifies as semantically
adjacent: `sanitation`↔`water` (4), `construction`↔`sanitation` (2),
`roads`↔`water` (2). When the model splits a complaint correctly but must then name
both buckets, it picks the wrong neighbour of a correct one.

## 5. Why the schema is not changed

The obvious repair is to split `sanitation`, and the evidence above says that is the
right long-term move. It is not done here because:

1. `ml/data/gold/annotation_schema.json` is frozen and shared with the gold
   annotation set. Changing the enum invalidates every metric in this repository,
   including the v2 baselines this audit exists to interpret.
2. The human annotators were not asked to relabel. Any new boundary would be
   invented by the same heuristic that produced the current error, so the "fix"
   would encode the same bias with more confidence.
3. `benefits` and `other` would need a labelling rule, not just an enum slot.

Recorded instead as a v3 precondition: the v3 plan measures a domain-accuracy floor of
0.82 rather than 1.0, explicitly leaving room for this ontology noise, and its
category K (8 cases) pairs domains that are confusable *by construction* so the
ceiling stays visible in future runs.

## 6. What a future split would have to decide

| question | why it is not answerable from this corpus |
| --- | --- |
| Does a fallen tree belong to `sanitation`, `roads`, or a new `greening`? | 88 % of tree complaints are filed as sanitation, so the corpus encodes the current answer, not the right one |
| Is a broken traffic light `electricity` or `roads`? | 22 rows total, 64 % sanitation — too few to fit, and the portal's answer is demonstrably wrong |
| Is a defective sidewalk `roads` or `sanitation`? | genuinely 59/31 split in the portal; no signal for which is right |
| What is `transport` for a subsidised bus pass? | the two source kinds are already merged, so the distinction was never labelled |
