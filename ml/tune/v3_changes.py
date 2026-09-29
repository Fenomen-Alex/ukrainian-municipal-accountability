"""The four v3 training-data changes (C1-C4), each traceable to a measurement.

Additive on purpose. ``ml/tune/build_dataset.py`` is left untouched so v2 stays
byte-reproducible; everything v3 changes lives here and is applied by
``ml/tune/build_v3.py``.

Every pattern below was mined from ``ml/data/{train,validation,test}.jsonl``
(6 837 records, 26 245 sentences of 12+ chars), so a reviewer can disagree with a
specific number rather than with "it seemed cleaner".

C1  boilerplate in ``issue``
    ``build_dataset._ADMIN_CLAUSE`` misses the single most common sentence in the
    corpus: ``Відповідь заявниці.`` (1 828 occurrences, 7.0% of sentences) -- a
    bare response-delivery stub. eval_v3's ``BOILER_SENT`` misses it too, for a
    spelling reason worth stating: ``заявник\\w*`` cannot match ``заявниці``,
    because Ukrainian replaces the ``к`` of "заявник" with ``ц``. The same gap
    silently drops ``заявнику``, ``заявника`` and every other case form.

C2  ``requested_action``
    Two separate defects.

    (a) The verb list is too small. Mining finds 283 sentences carrying an
        unambiguous explicit request that ``_REQ_VERBS`` misses, because it knows
        only a few polite forms. The dominant missing shape is the bare
        imperative/infinitive sentence opener -- ``Усунути порив мережі...``,
        ``Відновити асфальтне покриття...`` -- used 2 431 times (9.3% of
        sentences). The inventory below is the observed frequency list.

    (b) Worse, ``_derive_requested_action`` returns a whole *sentence* while
        ``_derive_issue`` returns the text *containing* that sentence, so
        ``action in issue`` holds by construction for every action the labeler
        ever emits. That is the mechanism behind the audit's 98.1%, and it is why
        no amount of verb-list tuning would have fixed it.

    The fix separates the fields: the action becomes the request *clause* (marker
    stripped, generic tail removed) while the issue keeps the restatement. Where
    the request is a content-free stub -- ``Прохання вжити заходи реагування.``
    alone occurs 441 times -- the action becomes empty, because a request to
    "take measures" names no action.

C3  terse upsampling
    ``plan.json`` says 178 usable terse kernels, ``short_note_analysis.md`` says
    928. Neither reproduces against the current ``_note_kernel``. Measured over
    the 841 train records under 150 characters: 246 have a kernel of at least 25
    characters (the analysis document's own usability criterion), 393 have a
    non-empty kernel, and 448 -- 53% -- have an *empty* kernel because the
    substance was filtered as boilerplate. 928 counts all 5 384 train rows, not
    the terse pool.

    So the pool is 246, not 178, and 8-12x is 1 968-2 952 examples, above the
    1 200-1 500 the design targeted (that target was 178x8 = 1 424).
    ``TERSE_UPSAMPLE = 8`` takes the bottom of the requested range.

C4  real share of the multi-topic augmentation
    The hard constraint, stated rather than hidden: ``build_multitopic.build_real``
    emits an example only when a hand annotation supplies >= 2 distinct kinds that
    are each verified as substrings of the record and span >= 2 domains. There are
    exactly 22 such records, and they are the only genuinely real multi-topic texts
    that exist. The corpus gives each record a single ``kind``, so widening the
    real pool means hand-labelling more complaints -- the subjective work the brief
    rules out.

    What C4 can do without inventing labels is rebalance how much of the *stream*
    the 22 real texts carry. The honest cost is recorded in the dataset report:
    reweighting 22 unique texts harder is memorisation pressure, not new
    information.
"""

from __future__ import annotations

import re

from ml.tune.build_dataset import _ADMIN_CLAUSE, _clean_text, _redact_pii

# --------------------------------------------------------------------------- C1
#: The delivery-mode phrase, euphony-aware: the corpus writes both the "в" and the
#: "у" form (584 vs 127 for "телефонн-"). ``[ву]\s+у?\s*`` matches either without
#: matching a bare "у" elsewhere. Shared by the R2 sentence and clause rules.
_R2_MODE = r"[ву]\s+у?\s*(?:телефонном\w*|письмов\w*)\s+режим\w*"

#: Sentences dropped **whole** by C1. A clause-level regex cannot do this safely:
#: the consent sentence is "Заявник надає згоду на обробку своїх персональних даних
#: та передачу їх третім особам відповідно до вимог ЗУ ...", and deleting any
#: substring of it leaves wreckage ("згоду на обробку своїх та передачу їх...").
#: A sentence that contains nothing but a consent, a delivery promise or a
#: signature is boilerplate in its entirety, so it is removed at sentence level.
C1_SENTENCE_DROP: tuple[tuple[str, str], ...] = (
    (r"(нада[єи]|згоду)\s+на\s+оброб|персональних\s+даних|"
     r"захист\w*\s+персональних", "data-protection consent sentence (2100+)"),
    (r"відповідь\s*(?:нада|надісл|переда|в\s+у?\s*(?:телефонн|письмов|електронн)|"
     r"по\s+телефону|телефоном|пошт|письмов|забере|отриман)",
     "response-delivery promise (687+543+172+129)"),
    (r"^\s*відповідь\s*[.!]?\s*$", "bare response stub"),
    (r"заявни\w*\s+(нада[єи]|повідомля[єи])", "identity clause (1029+962)"),
    (r"(з\s+повагою|з\s+поважанням|щоб\s+по\s+добр|з\s+вдячністю|"
     r"директор\w*\s+[А-ЯІЇЄҐ][а-яіїєґ]+\s+[А-ЯІЇЄҐ][а-яіїєґ]+)",
     "letter closing"),
    # R3: the generic closer sentence must be *only* the closer. Without the
    # end-anchor the pattern matched mid-sentence, so "Прохання вжити заходи та
    # усунути причину витоку води." was dropped whole (976 -> 852 sentences,
    # rescuing 124 that carry a concrete object after the generic head).
    (r"^\s*(прохання|просить|просимо|прошу)\s+(терміново|негайно)?\s*вжити\s+"
     r"(необхідні|відповідні|належні)?\s*заход\w*\s*(реагування)?\s*[.!]?$",
     "generic 'вжити заходи' closer, end-anchored (852)"),
    (r"просить\s+надати\s*[.!]?$", "Просить надати. (53)"),
    # R2: the phone-mode residue. "Відповідь заявниці в телефонному режимі."
    # survives the response-delivery sentence rule above because 'заявниці'
    # intervenes; its clause rule then strands 'в телефонному режимі'. A stranded
    # delivery-mode tail is boilerplate in its entirety, so sentence-level removal
    # (below) is correct; the retained stand-alone form is covered by
    # C1_CLAUSE_DROP so meaningful use ("некоректно спілкувалася в телефонному
    # режимі") is untouched.
    #
    # ``[ву]``, not ``в``: Ukrainian euphony alternates the preposition by the
    # preceding sound, and this corpus writes the "у" form 127 times against 584
    # "в" (and 3 vs 6 for "письмов-"). A ``в``-only rule silently left every "у"
    # record behind -- 123 single-stream survivors instead of the handful the
    # change was supposed to leave.
    #
    # A delivery sentence does not always put the mode phrase straight after
    # "Відповідь": the dominant surviving shape was "Відповідь заявниця бажає
    # отримати у телефонному режимі." (111 raw records), where the subject sits
    # between the head and the verb, so the response-delivery rule above (which
    # requires a verb/mode immediately after "відповідь") never fired, and clause
    # surgery then stranded a bare "у телефонному режимі". Two shape rules cover
    # the sentence as a whole instead: any sentence that *opens* with "Відповідь"
    # and contains the mode phrase, and the subject-first "Заявниця бажає
    # отримати відповідь … режимі". Together they catch 698 of the 710 raw mode
    # sentences; the residue is the two meaningful instrumentals (kept) and glued
    # period+capital boundaries (handled by the headless clause rule below).
    (r"^\s*відповідь\b[^.!?]*?" + _R2_MODE,
     "R2: 'Відповідь … в/у телефонному режимі' delivery sentence"),
    (r"^\s*заявни\w*\s+(?:баж\w+|проси\w+|хоч\w+)\s*отримати\s+"
     r"(?:відповідь\s+)?" + _R2_MODE,
     "R2: 'Заявниця бажає отримати відповідь … режимі' sentence"),
    (r"^\s*" + _R2_MODE + r"\s*[.!]?\s*$", "R2: stand-alone phone-mode strand"),
    (r"^\s*заявни(?:ку|ці|кам|цям)\s+(?:баж\w+\s+отримати\s+)?" + _R2_MODE +
     r"\s*[.!]?\s*$",
     "R2: dative-recipient delivery tail (…заявниці в/у телефонному режимі.)"),
)

C1_SENTENCE_RE = re.compile(
    "|".join(f"(?:{p})" for p, _ in C1_SENTENCE_DROP), re.IGNORECASE)

#: Clauses deleted from sentences that are *retained*, e.g. a consent tacked onto
#: the end of a real complaint. ``_ADMIN_CLAUSE`` is folded in so C1 is a strict
#: superset of the v1 behaviour.
C1_CLAUSE_DROP: tuple[tuple[str, str], ...] = (
    # "Надати відповідь і в телефонному режимі, і письмово" is a *clause*, and a
    # real issue can precede it in the same un-split segment ("…захват території
    # 40 см) Надати відповідь і в телефонному режимі, і письмово."), so dropping
    # the whole sentence loses the complaint. It must also precede the generic
    # "(надати|…)(відповідь|…)" rule below: at the same start position the first
    # alternative wins, and the generic one eats only "Надати відповідь", which
    # strands the conjunction ("…40 см) і").
    (r"надати\s+відповідь\s*(?:і\s+)?(?:[ву]\s+у?\s*)?"
     r"(?:телефонном\w*|письмов\w*)\s+режим\w*\s*,?\s*(?:і|та)?\s*письмов\w*",
     "R2: надати відповідь і в/у телефонному, і письмово (clause)"),
    (r"відповідь\s+заяв\w*", "Відповідь заявниці/заявнику/... (1828)"),
    (r"відповідь\s+надати", "Відповідь надати (687)"),
    (r"(надати|надіслати|передати)\s+(відповідь|телефоном|поштою|письмово|"
     r"електронн\w*\s+пошт\w*)", "надати відповідь телефоном/поштою (687+543)"),
    (r"(бажа[єи]\s+отримати)", "бажає отримати (191)"),
    (r"згоду\s+на\s+оброб\w*[^.!?]*", "згоду на обробку ... (trailing clause)"),
    (r"(заявни\w*|заявники)\s+повідомля\w*", "Заявниця повідомляє, що ... (12+)"),
    # R3: the generic closer is clause-final now. Without the guard the pattern
    # partial-eats "Просить вжити заходів" in "...вжити заходів для перенесення
    # туалету...", stranding the purpose clause the C2 purpose-rescue exists for.
    # Requiring punctuation/end after "заходи" means the closer drops only when it
    # names no object of its own.
    (r"(прохання|просить|просимо|прошу)\s+(терміново|негайно)?\s*вжити\s+"
     r"(необхідні|відповідні|належні|необхідних|відповідних|належних)?\s*"
     r"заход\w*\s*(реагування)?\s*(?=[.,!?;]|$)", "generic closer, clause-final"),
    # R2: delivery-mode clause residue inside a retained sentence. Only the
    # delivery frame is peeled ("просить надати в телефонному режимі" in a
    # sentence that also asks for something real); the guarded phone-mode strand
    # ("...спілкувалася в телефонному режимі", "...уточнити із заявницею в тел.
    # режимі") has no delivery verb before it and is untouched.
    (r"(?:відповідь\s+(?:заяв\w*\s*)?|просимо|просить)\s*надати\s+[ву]\s+у?\s*"
     r"(?:телефонном\w*|письмов\w*)\s+режим\w*",
     "R2: delivery-frame phone-mode clause"),
    (r"(?:надати|повідомити|передати)\s+[ву]\s+у?\s*"
     r"(?:телефонном\w*|письмов\w*)\s+режим\w*",
     "R2: надати/повідомити в/у телефонному режимі"),
    (r"листом\s+та\s+[ву]\s+у?\s*(?:телефонном\w*|письмов\w*)\s+режим\w*",
     "R2: листом та в/у телефонному режимі"),
    (r"(?:і|та)\s+[ву]\s+письмовом\w*\s*,\s*(?:і|та)\s+[ву]\s+у?\s*"
     r"(?:телефонном\w*|письмов\w*)\s+режим\w*",
     "R2: і в письмовому, і в телефонному режимі"),
    (r"[ву]\s+у?\s*телефонном\w*\s+режим\w*\s*,\s*(?:і|та)\s+письмов\w*",
     "R2: в телефонному режимі, і письмово"),
    (r"(?:надати|повідомити|передати)\s+(?:відповідь\s+)?заявни(?:ку|ці|кам|цям)\s+"
     r"[ву]\s+у?\s*(?:телефонном\w*|письмов\w*)\s+режим\w*",
     "R2: надати відповідь заявниці в/у телефонному режимі"),
    # R2: a glued period+capital boundary ("...продуктами.Відповідь заявниці у
    # телефонному режимі.") is not split by ``_SENT_SPLIT_C1``, so the sentence
    # rules above never see its delivery head; once that head is clause-deleted the
    # mode phrase is stranded at a boundary. Peel it only *after* sentence
    # punctuation, which leaves the meaningful mid-clause instrumental ("із
    # заявницею в телефонному режимі") untouched.
    (r"(?:^|(?<=[.;!?]))\s*" + _R2_MODE,
     "R2: stranded delivery-mode clause after a sentence boundary"),
    # R2: a phone-contact request ("Заявник просить зв’язатися у телефонному
    # режимі для надання конкретнішої інформації") is a delivery preference, not
    # complaint content, so the clause is peeled at a boundary (the purpose tail
    # goes with it, otherwise it is left dangling).
    (r"(?:^|(?<=[.;!?]))\s*заявни\w*\s+проси\w*\s+зв['’]?язатися\s+" + _R2_MODE +
     r"(?:\s+для\s+[^.!?]*)?",
     "R2: 'просить зв’язатися в/у телефонному режимі' contact request"),
)

C1_COMBINED = re.compile(
    "|".join([_ADMIN_CLAUSE.pattern] + [p for p, _ in C1_CLAUSE_DROP]),
    re.IGNORECASE,
)

#: A bare response stub left behind once its delivery clause is gone.
C1_RESPONSE_STUB = re.compile(
    r"(?:^|(?<=[.;!\s]))\s*відповідь\s*[.!]?\s*", re.IGNORECASE)

#: Deleting the consent clause strands the verb that introduced it: "...у мкр.
#: Черемушки та надати згоду на обробку персональних даних." leaves "...та надати".
C1_TRAILING_VERB = re.compile(
    r"(?:^|(?<=[.;!\s]))\s*(?:та|і|й|або|чи)\s+нада[ти]\s*[.!]?\s*$", re.IGNORECASE)

#: Removing a boilerplate clause can strand its tail: "Просить вжити заходи
#: реагування для відновлення води." loses its head and leaves "для відновлення
#: води." A complaint can never open with one of these function words, so a leading
#: one is proof the sentence was cut. "про" is deliberately excluded: "Про ремонт
#: дороги" is real content, whereas "для"/"та"/"і" only ever continue a clause.
C1_DANGLING = re.compile(
    r"^\s*(для|щоб|та|і|й|або|бо|проте)\s+[^.!?]{2,80}[,;:.]?\s*", re.IGNORECASE)

_SENT_SPLIT_C1 = re.compile(r"(?<=[.!?])\s+")


def clean_issue_c1(text: str) -> str:
    """C1: strip response-delivery, consent and identity boilerplate from ``issue``.

    Two stages, because they remove different things:

    1. Drop whole sentences that are *only* boilerplate. This is the stage that
       matters for the consent boilerplate, which is 2 100 of the 3 188 v2
       examples carrying boilerplate; no clause regex can remove it cleanly.
    2. Delete leftover clauses from the sentences that survive, then repair the
       punctuation and stranded fragments that deletion leaves behind.

    Pure deletion, like the v1 labeler it replaces: no rewording, so every
    surviving sentence is still the citizen's own.
    """
    kept = [s for s in _SENT_SPLIT_C1.split(_clean_text(text))
            if s.strip() and not C1_SENTENCE_RE.search(s)]
    out = " ".join(kept)

    for _ in range(3):
        new = C1_COMBINED.sub("", out)
        new = C1_RESPONSE_STUB.sub("", new)
        new = C1_TRAILING_VERB.sub("", new)
        if new == out:
            break
        out = new

    for _ in range(3):
        prev = out
        out = re.sub(r"\s+([.,;:!?])", r"\1", out)
        out = re.sub(r"([.!?])\s*([.!?])", r"\1", out)
        out = re.sub(r"[.,;:]{2,}", ".", out)
        out = re.sub(r"^\s*[.,;:!\-–—]+\s*", "", out)
        out = re.sub(r"\s*[.,;:!\-–—]+$", "", out)
        trimmed = C1_DANGLING.sub("", out).strip()
        if trimmed != out:
            out = trimmed
        if out == prev:
            break
    out = re.sub(r"\s{2,}", " ", out)
    return _clean_text(out)[:400]


# --------------------------------------------------------------------------- C2
#: Request verbs observed as a sentence opener, by corpus frequency. The six
#: highest cover 87% of explicit requests.
C2_VERBS: tuple[str, ...] = (
    "вжити", "надати", "відновити", "прибрати", "усунути", "відремонтувати",
    "провести", "замінити", "виконати", "забезпечити", "встановити", "вивезти",
    "перевірити", "вирішити", "зробити", "очистити", "опрацювати", "розглянути",
)

#: Politeness / request markers that may precede a verb in the same sentence.
C2_MARKER = re.compile(
    r"^\s*(прошу|прохаю|прохання|просимо|просить)\s*"
    r"(щонайменш|найшвидше|негайно|терміново|якомогашвидше)?\s*",
    re.IGNORECASE,
)

#: A request that names no object. "Take measures" is not an action.
C2_GENERIC = re.compile(
    r"^вжити\s+(необхідні|відповідні|належні|необхідних|відповідних|належних)?\s*"
    r"заход\w*\s*(реагування)?\s*[.!]?$",
    re.IGNORECASE,
)

_C2_VERB_RE = re.compile(r"^\s*(?:" + "|".join(C2_VERBS) + r")\b", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
#: A purpose clause rescues a generic verb: "...вжити заходи для відновлення води".
_C2_PURPOSE = re.compile(r"\s+для\s+\S+", re.IGNORECASE)


def extract_action_c2(text: str, max_len: int = 200) -> str:
    """C2: the request *clause*, or ``""`` when the request names no action.

    Returns a clause rather than a sentence so the field carries information the
    issue does not already state verbatim. Never invents a verb: every returned
    string is a literal substring of the input, minus markers and boilerplate.
    """
    for sent in _SENT_SPLIT.split(_clean_text(text)):
        stripped = C1_COMBINED.sub("", sent).strip()
        if not stripped:
            continue
        body = C2_MARKER.sub("", stripped)
        # An explicit request is either "<marker> <verb>..." or a bare
        # "<verb>..." sentence opener. Both are anchored at the start, so a verb
        # mentioned mid-sentence ("ми просили раніше") cannot be mistaken for a
        # request.
        if not _C2_VERB_RE.match(body):
            continue
        clause = body
        # A generic "вжити заходи" survives only if a purpose clause gives it an
        # object; otherwise there is no action to report.
        if C2_GENERIC.match(clause):
            purpose = _C2_PURPOSE.search(clause)
            if not purpose:
                return ""
            clause = clause[:purpose.start()].strip() + " " + purpose.group(0).strip()
        clause = re.sub(r"\s+", " ", clause).strip(" ,;:.!?").strip()
        if len(clause) < 4:
            return ""
        # ``_PERSON_NAME`` in the v1 redactor matches any capitalised word, so a
        # sentence-initial imperative ("Усунути порив мережі...") is eaten as if it
        # were a person's name. Redact the argument but never the verb, then put
        # the verb back -- it is what makes the field an action.
        verb = _C2_VERB_RE.match(clause).group(0).strip()
        tail = _redact_pii(clause[len(verb):]).strip()
        action = f"{verb} {tail}".strip() if tail else verb
        return action[:max_len]
    return ""


def action_is_whole_sentence(action: str, text: str) -> bool:
    """The defect C2 removes, kept as an explicit predicate so it stays measurable.

    ``action in issue`` is *not* the test: the issue legitimately restates the
    complaint, so a specific action stays a substring of it. What made the field
    useless was being a verbatim copy of a whole sentence.
    """
    if not action:
        return False
    return any(action == s.strip() for s in _SENT_SPLIT.split(_clean_text(text)))


# --------------------------------------------------------------------------- C3
#: Measured, not documented: see the module docstring. Of the 841 train records
#: under 150 characters, 246 have a note kernel of at least 25 characters.
TERSE_POOL_MIN_KERNEL = 246
TERSE_POOL_NONEMPTY = 393
TERSE_POOL_EMPTY = 448
TERSE_POOL_MIN_CHARS = 25
TERSE_POOL_MAX_CHARS = 150
TERSE_UPSAMPLE = 8

# --------------------------------------------------------------------------- C4
#: v2 weighted the 22 real multi-topic texts 12x against 3x for 957 synthetic, so
#: the 3 135-row augmentation was 8.4% real (264 of 3 135). C4 raises the real
#: weighting and lowers the templated one; the dataset report states the resulting
#: composition, because reweighting 22 unique texts is not the same as having more
#: real data.
MT_REAL_UPSAMPLE_V3 = 24
MT_SYNTHETIC_UPSAMPLE_V3 = 2


def terse_pool(records: list[dict], kernel_fn) -> list[dict]:
    """Train records under 150 chars whose note kernel is usable (>= 25 chars).

    ``kernel_fn`` is ``build_multitopic._note_kernel``. The >= 25 character rule
    is the usability criterion ``short_note_analysis.md`` states for itself; it
    is applied to the terse pool rather than to all train rows, which is where
    the documented 178/928/246 numbers differ.
    """
    out = []
    for r in records:
        text = r.get("content") or ""
        if len(_clean_text(text)) >= TERSE_POOL_MAX_CHARS:
            continue
        k = _clean_text(kernel_fn(text))
        if len(k) >= TERSE_POOL_MIN_CHARS:
            out.append(r)
    return out
