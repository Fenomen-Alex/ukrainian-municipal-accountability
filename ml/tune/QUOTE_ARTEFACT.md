# The literal ASCII double-quote artefact: root cause and fix

**Status: FIXED at the prompt boundary. No model was trained, re-fused or
modified. No benchmark expectation was changed.**

This note is the companion to `FINAL_MODEL_ASSESSMENT.md` §5 / P12, which
established *that* the quote failures existed. This note establishes *why*, and
records the pre-inference repair.

---

## 1. What the defect is

`parse_strict` (`ml/tune/serve_v2.py`) rejects a generation whose payload is not
strict JSON. On the two v3 arms the five affected cases failed with:

```
not strict JSON: Expecting ',' delimiter: line 1 column 126 (char 125)
```

That signature is an unescaped `"` closing an `issue` string early. The v3 arms
copy the complaint span into the output verbatim, so a literal `"` in the input
becomes a literal `"` inside a JSON string, and the object stops parsing. In the
committed 960-row probe this is exactly 50/50 failures on the two v3 arms
(5 cases x 2 arms x 5 settings), all at the same offset, while v2 parses 25/25.

## 2. Two independent paths, one consequence

### 2.1 The eval/data path manufactures an empty pair

`ml/tune/build_dataset.py` `_redact_pii` removes personal data using
`_PERSON_NAME`, which matches any capitalised 3+ letter word. It therefore
over-matches organisation, shop and utility names, and it deletes the *content*
of a quoted name while leaving the enclosing marks:

```
source:  працівники "Екостайлу" не відповідають
    ->  працівники "" не відповідають
```

The empty pair is then copied into the `eval_v3` inputs and into the training
**targets**, where `ml/tune/residue` counts 12,528 occurrences in the v2 target
set and 15,274 in v3.2 (a single repeated example). That is the "learned
copying" the final assessment recorded — the model was explicitly taught to emit
`""`.

### 2.2 The production path passes real quotes straight through

`serve_v2.build_prompt` previously handed the complaint to the model verbatim:
no `_redact_pii`, no `_clean_text`, no quote handling at all. A quoted shop or
utility name in live traffic therefore reached the model as a raw `"`.

So the hazard in production is **not** manufactured — it is a legitimate
quotation that JSON cannot carry verbatim.

## 3. Why the fix is normalisation, not deletion

Deleting the marks would destroy real quotation semantics (`"Копілка"` names a
shop) and would hide the defect rather than remove it. Post-generation regex
repair was ruled out: it would mask model behaviour and touch the serving
contract's strictness.

Instead the character is mapped onto typography the corpus already uses, so the
existing weights already handle it and **no retraining is required**. The corpus
census (`train`/`validation`/`test`, 112 ASCII quotes in 55 records) shows the
quote is doing exactly two jobs:

| role | example | count | normalisation |
|---|---|---|---|
| mid-word apostrophe typo | `під"їзду`, `роз"яснень` | 12 | U+2019 `’` |
| quotation delimiter | `КП "ЖЕО №2"`, `"Копілка"` | 100 | U+201C `“` / U+201D `”` alternating |

No measurement, unit or other semantic use of `"` occurs anywhere in the corpus.
`“…”` is already the second most common quotation form (~400 pairs) and `«…»`
appears 234 times, so both candidates are in-distribution; `“”` was chosen
because its opener is unambiguous, unlike `’` which is also the apostrophe.

The transform deletes nothing, so length and every non-quote character are
preserved, and it is idempotent.

## 4. Where it is applied, and where it deliberately is not

`normalize_quotes` lives in `ml/tune/build_dataset.py` and is applied in
`serve_v2.build_messages`. That is the single boundary every inference path
crosses — `run_eval_v3`, `decoding_probe`, `behavior_v3`, `verify_v2` and
`serve_v2` itself all render through `build_prompt` — so production and all
evaluation are covered by one edit.

It is **not** applied inside `_clean_text`. That cleaner feeds the v2 / v3 /
v3.2 corpus builders whose artefacts are frozen and SHA-pinned; normalizing
there changes those artefacts. An earlier attempt to do exactly that was caught
by `ml/tests/test_v3_2.py::test_rebuild_is_deterministic_and_hits_the_pinned_sha`
and reverted. The prompt boundary is both the earliest *and* the only place that
does not mutate a frozen artefact.

## 5. Affected records

| corpus | ASCII quotes | records | mid-word |
|---|---|---|---|
| `ml/data/train.jsonl` | 96 | 47 | 12 |
| `ml/data/validation.jsonl` | 14 | 7 | 0 |
| `ml/data/test.jsonl` | 2 | 1 | 0 |
| **total** | **112** | **55** | **12** |

`eval_v3` has 146 cases; **6** carry an ASCII quote, and after normalization
**0** do:

| case | kind |
|---|---|
| `ev3-016`, `ev3-024`, `ev3-032`, `ev3-046`, `ev3-052` | redaction-created empty `""` |
| `ev3-123` | legitimate `"Копілка"` — already parsed, kept as a regression guard |

The five empty-pair cases are exactly the set `ml/tune/residue` reports, because
its detector is `""`-only. The report's "5 cases" and this note's "6 ASCII
quotes" are the same finding measured at different strictness.

The 6 `expected_topics` entries that contain an ASCII quote are **benchmark
references and were left untouched**. They will not string-match the `“…”` the
model now emits; that is the correct trade, since editing expectations to match
a fix would destroy the benchmark's value.

## 6. Before / after, on real generations

Same loaded adapters, canonical `build_prompt`, probe `budget1200` config
(`temp=0.0, top_p=0.0, max_tokens=1200`). The only difference between the two
columns is whether `normalize_quotes` ran. The "before" column reproduces the
committed probe exactly, including the `char 125` offset.

| case | corrected-v3 before -> after | v3-2 before -> after | v2 before -> after |
|---|---|---|---|
| `ev3-016` | JSONFAIL -> **VALID** | JSONFAIL -> **VALID** | VALID -> VALID |
| `ev3-024` | JSONFAIL -> **VALID** | JSONFAIL -> **VALID** | VALID -> VALID |
| `ev3-032` | JSONFAIL -> **VALID** | JSONFAIL -> **VALID** | VALID -> VALID |
| `ev3-046` | JSONFAIL -> **VALID** | JSONFAIL -> **VALID** | VALID -> VALID |
| `ev3-052` | JSONFAIL -> **VALID** | JSONFAIL -> **VALID** | VALID -> VALID |

**5/5 repaired on each v3 arm; v2 unaffected (no regression).** The pre-fix error
on every one of the ten is `Expecting ',' delimiter ... char 125` — the quote
signature — and it disappears entirely.

## 7. `ev3-083` — separate defect, not fixed here

`ev3-083` contains no ASCII quote, `normalize_quotes` returns it byte-identical,
and it fails identically before and after (`Unterminated string`). It is the
decoding-sensitive case already documented in the final assessment: corrected-v3
fails it in 2/5 settings and v3.2 in 0/5. It is **not** addressed by this change
and must not be credited to it.

## 8. Production safety

* Text with no ASCII quote produces a **byte-identical** prompt, so the v2
  serving contract is unchanged for the overwhelming majority of traffic.
* `DEFAULT_MODEL`, `DEFAULT_MAX_TOKENS`, `DEFAULT_TEMPERATURE`, `THINK_PREFIX`
  and `SYSTEM_PROMPT` are all unchanged and pinned by tests.
* The system prompt keeps its JSON schema quotes — only the *user* turn is
  normalized.
* No post-generation repair: strict parsing stays strict, so any other malformation
  still surfaces instead of being silently cleaned.

## 9. Validation

```
.venv/bin/python -m pytest ml/tests/ -q                       # 429 passed, 3 skipped
.venv/bin/python -m ml.tune.verify_final_assessment           # OK (380/380)
.venv/bin/python -m ml.tune.verify_v3_2_report                # PASS 197/197
.venv/bin/python -m ml.tune.artifact_sha --verify              # ALL REFERENCE ARMS UNCHANGED
```

`ml/tests/test_quote_normalization.py` holds 24 gates: the five repairs, the
legitimate-quotation guard, idempotence, no-character-but-quotes changed,
determinism, `None` passthrough, prompt byte-identity, and the pinned serving
constants.