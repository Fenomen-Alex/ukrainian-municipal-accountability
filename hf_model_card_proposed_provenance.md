# Proposed HF model-card correction — metadata only

Prepared before any change to the public model card. **No weight, config,
tokenizer, chat-template, or quantization change is proposed or permitted.**

## Target

| | |
|---|---|
| Repo | `Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b` |
| File | `README.md` (model card), metadata only |
| Revision under edit | `main` |
| Published card before edit | SHA-256 `837fdd100a474b7de0648d1a331bb80b653021a46f32d42aaa9a9165bf0ba500`, 9998 bytes |
| Section | `## License and data provenance` — **paragraph 2 only** |

Paragraph 1 of that section (Apache-2.0 weights, base model, MLX 4-bit
conversion) is accurate and is **left byte-identical**. Only the paragraph below
is replaced.

## The false statement being removed, verbatim

> The **training data is not redistributed here and its license is not
> documented** in the source project. The municipal complaint corpus has no
> recorded source URL, license, or redistribution terms. Only the derived weights
> are published. Anyone redistributing or building on this checkpoint should
> confirm the upstream data terms independently.

## Exact proposed replacement paragraph

> The **source data is CC BY 4.0 International** and is documented, not
> undocumented. It is the Kropyvnytskyi City Council open dataset
> «Дані про надходження звернень на телефонні "гарячі лінії", в
> аварійно-диспетчерські служби, телефонні центри тощо», published by the
> Executive Committee of the Kropyvnytskyi City Council at
> <https://data.kr-rada.gov.ua/dataset/1770a320-880f-4cd6-9840-78b9e8d93553>.
> The publisher's dataset metadata declares `license_id: cc-by` with title
> "Creative Commons Attribution", and the portal's default licensing is
> CC BY 4.0 International unless otherwise stated. That licence permits
> redistribution and re-use **on the condition that the creator is appropriately
> credited**.
>
> The required attribution and the exact source URL, resource identifier,
> licence wording and verification steps are recorded in the source project's
> `DATA_PROVENANCE.md`.
>
> **Two distinct licences apply.** The **model weights and source code are
> Apache-2.0**. The **source dataset is CC BY 4.0 International**, and the CC BY
> terms attach to the data, not to the weights. The derived weights are published
> under Apache-2.0.
>
> The source project's public repository **does** track source-derived corpus
> files. Those corpora retain rows containing phone numbers and other contact
> details, and the project's redaction step is known to be incomplete; they are
> provided under CC BY 4.0, not under the Apache-2.0 terms. Anyone redistributing
> or building on this checkpoint should read the provenance document before
> redistributing any data.

## Fact check of every claim above

| Claim | Basis | Verified |
|---|---|---|
| CC BY 4.0 International | portal default + `license_id: cc-by` | CKAN `package_show`, HTTP 200 |
| Dataset title / publisher | dataset metadata | CKAN `package_show` |
| Official source URL | dataset page | HTTP 200 |
| "permits redistribution ... appropriately credited" | Open Definition text quoted by the portal | `opendefinition.org/licenses/cc-by` |
| Weights + code Apache-2.0 | HF frontmatter `license: apache-2.0` | confirmed |
| Corpus retains phone numbers | measured scan, 187 phone-like rows | reproduced from HEAD |
| Redaction incomplete | `_redact_pii` over-match, 114 email-like rows | reproduced from HEAD |
| Corpora tracked publicly | `git ls-files`, 25 JSONL | reproduced from HEAD |

## Deliberately NOT claimed

The replacement does **not** state that raw complaint rows are absent from the
public repository, because they are not. No claim is made that the data is
suitable for redistribution beyond what CC BY itself permits, and no legal
conclusion is drawn about personal data.