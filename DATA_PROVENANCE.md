# Data provenance and licensing

Authoritative record of where the training and evaluation corpora came from and
what their terms actually are. Every statement here was verified against the
publisher's own machine-readable metadata; the commands are given so the result
can be re-derived.

**Bottom line:** the source corpus is published by the Kropyvnytskyi City Council
under **Creative Commons Attribution 4.0 International**, which **permits**
redistribution and re-use **provided the creator is credited**. That attribution
condition was **not satisfied** in this repository before this document existed,
and is now recorded here.

This document covers **data**. Model-weight licensing is separate — see
[the distinction below](#data-terms-vs-model-licensing).

## Source dataset

| Field | Value |
|---|---|
| Title (original) | «Дані про надходження звернень на телефонні "гарячі лінії", в аварійно-диспетчерські служби, телефонні центри тощо» |
| Title (English) | Data on the receipt of appeals to telephone "hotlines", emergency-dispatch services, telephone centres, etc. |
| Publisher | «Виконавчий комітет Кропивницької міської ради» — Executive Committee of the Kropyvnytskyi City Council (org slug `vykonavchyi-komitet-kropyvnytskoi-miskoi-rady`) |
| Portal | `https://data.kr-rada.gov.ua` — "Портал відкритих даних Криворізької громади" |
| Dataset id | `9109c3a3-980f-45d0-a83e-b60f87d583d4` |
| Dataset slug | `1770a320-880f-4cd6-9840-78b9e8d93553` |
| Maintainer | Колесник Дарина Сергіївна — `data@krmr.gov.ua` |
| Metadata created | `2023-03-30T04:55:16` |
| Metadata modified | `2026-10-01T08:07:15` |
| Resource | id `ddf7ffe9-d8e0-4646-93b2-b10d0b4bfe63`, name `appeals.csv`, `text/csv`, 15,727,208 bytes |
| Resource created | `2026-02-10T14:32:36` |
| Resource modified | `2026-10-01T08:06:42` (ETag `"1790842002.236968-15727208-2380993357"`) |

Official dataset page:
<https://data.kr-rada.gov.ua/dataset/1770a320-880f-4cd6-9840-78b9e8d93553>

Resource download (current):
<https://data.kr-rada.gov.ua/dataset/9109c3a3-980f-45d0-a83e-b60f87d583d4/resource/ddf7ffe9-d8e0-4646-93b2-b10d0b4bfe63/download/appeals.csv>

Dataset description, as published: «Набір містить дані про реєстраційний номер,
дату й час надходження, тип звернення, тематичну категорію, адресу,
географічні координати, статус, виконавця та результати розгляду Кропивницькою
міською радою» — registration number, date/time of receipt, appeal type,
thematic category, addresses, geographic coordinates, status, executor, and the
outcome of review by the City Council.

### Two corrections to the project's earlier source reference

1. **The dataset id in earlier project documentation was wrong by one
   character.** It recorded `9109c3a3-980f-45d0-a83e-b60f87d583**c4**`; the real
   id ends `…583**d4**`. The incorrect id returns HTTP 404 from the publisher's
   API. The resource id `ddf7ffe9-…` recorded by the project is correct.
2. **The recorded filename no longer exists.** The project documented
   `appeals_2026-08-01.csv`. The portal exposes a **single mutable resource**
   named `appeals.csv`, created 2026-02-10 and rewritten in place (last modified
   2026-10-01). **There is no immutable, dated snapshot and no published
   checksum for the source.** Any future re-pull will not be byte-comparable to
   an earlier pull.

## License — exact published wording

Stated in the dataset's own metadata (`license_id: cc-by`):

> `license_title`: **Creative Commons Attribution**
> `license_url`: `http://www.opendefinition.org/licenses/cc-by`

The dataset page additionally declares Open Definition conformance:

> «Цей набір даних відповідає вимогам Open Definition» — "This dataset conforms
> to the Open Definition requirements"
> (`<a href="http://opendefinition.org/okd/" title="Цей набір даних відповідає вимогам Open Definition">`)

The portal footer states a site-wide default licence:

> «Весь контент доступний за ліцензією **Creative Commons Attribution 4.0
> International license**, якщо не зазначено інше» — "All content is available
> under the Creative Commons Attribution 4.0 International licence, unless
> otherwise specified."

The Open Definition page linked by the publisher (`cc-by`) states:

> "The Creative Commons Attribution license allows re-distribution and re-use of a
> licensed work on the condition that the creator is appropriately credited."

It lists the applicable full texts as CC BY 4.0 / 3.0 / 2.5 / 2.0 / 1.0, links
`https://creativecommons.org/licenses/by/4.0/` as the overview, and notes
**"Domain of Application: Content, Data (latter 4.0 only)"** — i.e. for
*data*, the 4.0 version is the applicable one.

**Stated version:** the portal's `license_id` field carries no version string.
The dataset-level link points at the Open Definition `cc-by` entry whose overview
is **CC BY 4.0**, and the portal-wide default is explicitly **CC BY 4.0
International**. For a dataset, CC BY 4.0 is therefore the applicable grant. The
dataset-level resource record carries **no** `license_id` of its own — the terms
come from the dataset and the portal default, not from the file entry.

### Redistribution terms, attribution requirements, derivatives

| Question | Answer from the source text |
|---|---|
| Redistribution permitted? | **Yes** — CC BY expressly "allows re-distribution and re-use". |
| Re-use / derivatives permitted? | **Yes** — "re-use of a licensed work" is allowed. |
| Attribution required? | **Yes** — "on the condition that the creator is appropriately credited". The publisher's own guidance: include a link to or a full copy of the licence, and a notice such as "This work is licensed under a Creative Commons Attribution [version] License." |
| Commercial use? | Not restricted by CC BY. *(Recorded as the licence text stands; no additional portal restriction was found.)* |
| ShareAlike / copyleft? | **No** — CC BY is permissive, not copyleft. |
| Derivatives / ML training specifically addressed? | **Not addressed.** No dataset-level `extras` are published, and nothing on the page mentions training, models or derivatives. Derivation is nonetheless covered by the general "re-use" grant. |
| Additional restrictions? | **None found.** |

### Attribution owed by this project

CC BY conditions redistribution on crediting the creator. This project therefore
owes the following, and this document is the notice:

* **Creator:** Executive Committee of the Kropyvnytskyi City Council
  («Виконавчий комітет Кропивницької міської ради»)
* **Title:** «Дані про надходження звернень на телефонні "гарячі лінії", в
  аварійно-диспетчерські служби, телефонні центри тощо»
* **Source:** <https://data.kr-rada.gov.ua/dataset/1770a320-880f-4cd6-9840-78b9e8d93553>
* **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Modifications:** the source rows were reformatted to JSON, split into
  train/validation/test partitions, passed through a heuristic PII redaction and
  weak-label heuristic, and converted into chat-formatted training examples.
  The corpora are **modified**, not verbatim copies.

`data.gov.ua` was also checked: sibling "hotline appeals" datasets from roughly
20 other municipalities are syndicated there under the same CC-BY label, but the
Kropyvnytskyi dataset itself is **not** present on the national portal. The
publisher's own portal above is the authoritative source.

## Privacy and open-data notes — none published

The publisher publishes **no** privacy, personal-data, confidentiality or GDPR
note for this dataset. The page contains zero occurrences of privacy,
«персональні», «конфіденційність» or GDPR terms. Nothing addresses whether the
open-data licence extends to personal data.

Yet the resource demonstrably contains personal data. Measured on a 787-row
sample of the live CSV (HTTP 206 range request, 600 KB):

* **19.1 %** of rows contain a phone-like number in the free-text `content`
  field.
* **13.0 %** of rows carry a specific building number
  (`addressLocatorBuilding`).
* 20 columns, including free-text `content`, street
  (`addressThoroughfare`) and building identifiers, plus the registration
  number `uid`.

So the corpus is published as open data under CC BY while containing personal
data, with no accompanying privacy caveat. This is a fact about the source, not
a legal conclusion about it.

## What this repository redistributes

GitHub distribution is **not** limited to code. Of 216 tracked files, **25 are
`.jsonl` corpora totalling 85.6 MB**, and they retain the source's verbatim
20-column schema (`content`, `CATUTTC`, `addressLocatorBuilding`,
`addressThoroughfare`, `uid`, …). No `.csv` from the source and no model weights
are tracked; adapters are gitignored.

Reproduce everything in this section with:

```bash
python -m ml.tune.audit_public_data          # table
python -m ml.tune.audit_public_data --write  # refresh the manifest
python -m ml.tune.audit_public_data --check  # assert the manifest is current
```

The machine-readable record is `ml/data/tune/public_data_manifest.json`.

### Corpus classes

25 tracked paths resolve to **24 distinct files** (one is a symlink) and **21
distinct content hashes** — several corpora are byte-identical copies, so row
counts per path would double-count. Totals: **85.6 MB, 32,888 rows per path /
30,291 distinct rows**.

| Class | Files | MB | Rows (per path) | Phone-like | Email-like | Verbatim source text | Needed for offline tests | Intended public |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_derived_corpus` | 12 | 83.19 | 29,897 | 210 | 118 | **yes** | **yes, 48 tests fail without it** | **undecided — see below** |
| `annotation_or_provenance_metadata` | 3 | 0.95 | 1,725 | 10 | 3 | partial (uids, quoted text) | yes | undecided |
| `generated_model_output` | 9 | 1.24 | 1,120 | 0 | 0 | no (model output) | yes | yes |
| `benchmark_case_set` | 1 | 0.26 | 146 | 1 | 0 | yes (complaint text) | yes | undecided |

Byte-identical duplicate groups (rows counted once above):

| SHA-256 prefix | Paths |
|---|---|
| `49a7c3518558` | `ml/data/tune/v2/test.jsonl` == `ml/data/tune/v3/treatment/test.jsonl` |
| `c7cea07ec00f` | `ml/data/tune/v2/validation.jsonl` == `ml/data/tune/v3/treatment/validation.jsonl` |
| `fb553d57a4ee` | `ml/data/tune/smoke/results/finetuned-v2/smoke_results.jsonl` == `.../finetuned/smoke_results.jsonl` |
| — | `ml/data/tune/v2/valid.jsonl` is a **symlink** to `validation.jsonl` |

Aggregate PII indicator across all tracked corpora: **221 phone-like rows,
121 email-like rows** counted per path. These are heuristic pattern hits, not a
privacy audit, and they support no legal conclusion.

### Classification of the repository data-distribution state

> **C — distribution status cannot be responsibly established from the available
> evidence.**

The three permitted classifications were tested against evidence rather than
chosen by preference:

**A is not supportable.** No document in this project's history records the owner
reviewing or accepting the PII exposure. The opposite is on record: until commit
`15a44be` every document asserted the source licence was *unknown*, so the
question of CC BY *plus* live contact details was never put to anyone.

**B is the correct direction but its remedy is unavailable.** The corpora are not
required to reproduce the published artifact — the HF weights plus
`ml/data/tune/artifact_sha256.json` fully determine and verify v2, and model
development is frozen — so they are not *necessary* to be public. But removing
them from the public repository is not achievable by the mechanism B implies:

1. **The data is already permanently public.** The corpora were committed in
   `e217197`, confirmed an ancestor of `origin/master`. `git rm --cached` only
   untracks them going forward; every row stays downloadable from public
   history. Purging it requires a history rewrite
   (`git filter-repo` + force-push), which destroys shared history and needs
   explicit owner authorisation.
2. **Untracking breaks the test suite, badly.** Measured at `15a44be`: hiding the
   12 `source_derived_corpus` files gives **48 failures, 419 passed, 1 collection
   error** (`ml/tests/test_tune_dataset.py` fails at *collection* time). Removing
   them is not a quiet `git rm`; it requires skip guards in
   `test_tune_dataset.py`, `test_v2_augmentation.py` and `test_v3_changes.py`,
   and it would leave the suite unable to validate corpus construction at all.

So B cannot be executed as written, and executing its cosmetic form would
misrepresent the exposure as fixed while changing nothing that matters.

**Therefore the question stays open pending exactly one owner decision.** Either
outcome closes it:

| Owner decision | Effect | Cost |
|---|---|---|
| Formally accept the exposure (records A) | classification becomes **A**, distribution may proceed | personal data of Ukrainian citizens stays publicly redistributable under CC BY, which is a copyright licence and not a privacy clearance |
| Authorise a history rewrite (records B) | classification becomes **B** | `git filter-repo` over `e217197` + force-push, breaking clones and SHAs; plus skip guards for 3 test files |
| Neither | classification stays **C** | distribution stays **BLOCKED** |

Nothing was removed to improve this audit's appearance. The tracked corpora are
unchanged; the remediation above is documented, not applied.

## Preprocessing and PII handling

The pipeline applies `_redact_pii()` in `ml/tune/build_dataset.py`. Its actual
behaviour, verified directly:

| Input | Output |
|---|---|
| `Заявнику надано номер телефону ГЛ Пенсійного фонду (0800503753).` | ` надано номер телефону ГЛ  фонду ().` — number stripped |
| `Звернувся Іван Петренко, тел. 066 672 64 30, email ivan.p@example.com` | ` , тел. , email ivan.p@example.com` — name and phone stripped, **email retained** |
| `Біля маг. "Копілка" тече каналізація` | ` маг. "" тече каналізація` — **shop name destroyed, empty quotes created** |

Two defects are visible in that table and both are load-bearing:

1. **Redaction is incomplete.** It is pattern-based and does not catch every
   phone or any email, which is why 187 phone-like and 114 email-like rows
   survive in tracked corpora.
2. **`_PERSON_NAME` over-matches**, so legitimate names are destroyed. The third
   row is the direct cause of the empty-quote JSON failures analysed in
   `ml/tune/QUOTE_ARTEFACT.md`; it is reproducible in one line.

Fixing either would require rebuilding the corpora and retraining, which is out
of scope: model development is **frozen**.

## Data terms vs model licensing

These are different grants and must not be conflated.

* **Model weights:** Apache-2.0, declared in the Hugging Face model card
  frontmatter (`license: apache-2.0`) and in the repository's model card.
  Verified present in the published artifact. This grant covers **the weights**.
* **Source corpus:** Creative Commons Attribution 4.0 International, per the
  publisher. Conditions on redistribution.

The weights are a **derived work** of a CC BY corpus. The Apache-2.0 licence on
the weights does **not** relicense the corpus, does not discharge the CC BY
attribution condition for the corpus, and must not be read as applying to it.
Apache-2.0 also does not oblige anyone to attribute a CC BY corpus, so the two
obligations have to be satisfied separately — which is why this document exists
alongside the model card.

## Known correction needed on the public model card

**Not applied** — the public Hugging Face artifact was deliberately not modified.
The published card currently states:

> "The **training data is not redistributed here and its license is not
> documented** in the source project. The municipal complaint corpus has no
> recorded source URL, license, or redistribution terms."

That statement was **factually wrong** on both counts: a source URL is recorded
above, and the licence is CC BY 4.0. The card understated the available grant and
omitted the attribution condition.

**Corrected.** The false paragraph was replaced with a metadata-only edit on
`main`; weights, config, tokenizer, chat template and quantization were not
touched. The exact before/after text and the fact-check of every new sentence are
recorded in `hf_model_card_proposed_provenance.md`. The corrected card now states
the dataset, the official source URL, the CC BY 4.0 terms and attribution
condition, the licence split between Apache-2.0 weights/code and CC BY 4.0 data,
and that the source project **does** track source-derived corpora which may retain
contact details.

Everything else on the card is accurate: model identity (v2 / attempt-10),
Apache-2.0, base model `mlx-community/Qwen3-8B-4bit`, 4-bit group-size-64
quantization, intended use, byte-exact system-prompt requirement, serving
assumptions, Apple-Silicon-only constraint, repository link, and an honest
limitations section. The card's limitations do not mention the `ev3-083` decoding
defect or the `_PERSON_NAME` over-matching recorded in this repository.

## Unresolved issues

1. **The source is not versioned.** A single mutable `appeals.csv` with no
   published checksum means this repository's corpora cannot be re-derived or
   proven against the source later. Any regeneration would produce a different
   corpus.
2. **No licence version string** in the `license_id` field; CC BY 4.0 is inferred
   from the Open Definition overview and the portal-wide default.
3. **No privacy terms** published by the publisher despite measurable personal
   data in the resource.
4. **The data-distribution decision is undecided — classification C.** 221
   phone-like / 121 email-like rows are publicly distributed via GitHub, the data
   is permanently in public history at `e217197`, and the decision to accept or
   remove it belongs to the owner. This is the **sole remaining release blocker**.
5. Residual `_redact_pii` incompleteness, unchanged and documented above.

## Reproducing this investigation

```bash
# dataset metadata (note: package_show needs the slug, not the id)
curl -s "https://data.kr-rada.gov.ua/api/3/action/package_show?id=1770a320-880f-4cd6-9840-78b9e8d93553"

# the dataset id as previously recorded in this project -> 404, proving the typo
curl -s "https://data.kr-rada.gov.ua/api/3/action/package_show?id=9109c3a3-980f-45d0-a83e-b60f87d583c4"

# publisher + licence page
curl -sL "http://www.opendefinition.org/licenses/cc-by"

# resource reachability and size
curl -sI "https://data.kr-rada.gov.ua/dataset/9109c3a3-980f-45d0-a83e-b60f87d583d4/resource/ddf7ffe9-d8e0-4646-93b2-b10d0b4bfe63/download/appeals.csv"
```