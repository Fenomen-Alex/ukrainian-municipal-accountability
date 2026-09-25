"""Build chat-format MLX training data for Ukrainian municipal complaints -> structured JSON.

The target schema is the frozen gold annotation schema (ml/data/gold/annotation_schema.json).
Labels are WEAK: every field is derived by a pure, deterministic, unit-testable transform
over the existing record. No synthesis, no paraphrasing, no invented values: when a field
cannot be derived it is left empty and the schema's honesty rules keep the model from
inventing it.

Output: ml/data/tune/{train,valid,test}.jsonl in MLX-LM ChatDataset format
({"messages": [...], "uids": [...]}).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "annotation_schema.json"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = DATA_DIR / "tune"

# --- kind (weak source label) -> schema domain enum -------------------------

KIND_TO_DOMAIN: dict[str, str] = {
    "Будівництво та ремонт доріг, вулиць": "roads",
    "Будівництво, містобудування, архітектура": "construction",
    "Гаряче та холодне водопостачання": "water",
    "Діяльність органів місцевого самоврядування": "government",
    "Експлуатація та ремонт житла ( у т.ч. ліфтів, сантехнічного обладнання тощо)": "housing",
    "Електропостачання населених пунктів, будинків": "electricity",
    "Питання пов’язані з торгівлею (у тому числі стихійна торгівля)": "commerce",
    "Плата за житло та комунальні послуги ( у т.ч. підвищення тарифів)": "payments",
    "Пільгове перевезення пасажирів": "transport",
    "Робота пасажирського транспорту (у т. ч. електричного транспорту)": "transport",
    "Санітарний стан, благоустрій населених пунктів, прибудинкових територій": "sanitation",
    "Теплопостачання": "heating",
}

_REQ_VERBS = re.compile(
    r"\b(прош(у|имо|уємо)|проханн\w*|вимага\w*|просимо надати|зобов’яжіть|"
    r"прошу вжити|просимо вжити|надайте|проведіть|відремонтуйте|встановіть|"
    r"приберіть|забезпечте|здійсніть|усуньте)\w*\b",
    re.IGNORECASE,
)

# Docket/admin boilerplate clauses that are not part of the citizen's complaint:
# consents, "відповідь надати ...", "заявник/заявниця" identity clauses.
_ADMIN_CLAUSE = re.compile(
    r"(,?\s*(заявни(к|ця)|заявники)\s*(надає|надають|повідомляє|повідомляють|"
    r"не згодна|не згодні|скаржиться|скаржаться)[^.]*\.)|"
    r"(,?\s*(відповідь|відповіді)\s*(надати|надавати)[^.]*\.)|"
    r"(,?\s*згоду\s+(надає|надають)[^.]*\.)|"
    r"(,?\s*роз’ясненн(я|ня)[^.]*\.?)",
    re.IGNORECASE,
)

_STREET_WORD = r"(?:\d+)?[А-ЯІЇЄҐ][а-яіїєґ0-9’'-]*"
_STREET_PHRASE = re.compile(
    rf"(?P<stem>\b(?:вул\.|вулиц[яі]|провул(?:ок|ку|ком)?\.|просп(?:ект)?\.|"
    rf"б-р\.|бул\.|пл\.|шосе|дорога))\s*(?P<name>{_STREET_WORD}(?:\s+{_STREET_WORD})*)",
    re.IGNORECASE,
)
_PERSON_NAME = re.compile(
    r"(?P<name>(?:[А-ЯІЇЄҐ][а-яіїєґ’'’-]{2,})(?:\s+[А-ЯІЇЄҐ]\.?\s*[А-ЯІЇЄҐ]\.?)?)"
)
_PHONE = re.compile(r"(?:\+?38)?0\d{2}[-\s)]?\d{3}[-\s]?\d{2}[-\s]?\d{2}")
_BUILDING_NO = re.compile(r"\b(?:будинок|буд\.|№|корп\.?)\s*[№]?\s*(\d+[а-яА-Я]?)", re.IGNORECASE)


def _street_spans(text: str) -> list[tuple[int, int]]:
    """Text spans of street phrases, used to protect street names from redaction."""
    spans = []
    for m in _STREET_PHRASE.finditer(text):
        stem = m.group("stem")
        name = m.group("name")
        # name may trail with "буд.", "корп.", digits; keep the stem+name tokens.
        end = m.start("stem") + len(stem) + len(name.rstrip("., "))
        spans.append((m.start("stem"), min(end, len(text))))
    return spans


def _redact_pii(text: str) -> str:
    """Remove obvious personal data (initialized names and phone numbers) from free text.

    Street-name spans are protected so 'вул. Леоніда Каденюка' is not stripped.
    Deterministic and auditable; not a full PII detector.
    """
    if not text:
        return text
    spans = _street_spans(text)
    protected = [False] * len(text)
    for a, b in spans:
        for i in range(a, b):
            protected[i] = True

    replacements = []
    for m in _PERSON_NAME.finditer(text):
        if any(protected[i] for i in range(m.start(), m.end())):
            continue
        # keep short words (single-token 'вул.', abbreviations) — name group is 2+ letters
        replacements.append((m.start("name"), m.end("name")))
    for m in _PHONE.finditer(text):
        replacements.append((m.start(), m.end()))

    out = []
    last = 0
    for a, b in sorted(replacements):
        out.append(text[last:a])
        out.append("")
        last = b
    out.append(text[last:])
    return "".join(out)


@dataclass
class LabeledRecord:
    uid: str
    domain: str
    issue: str
    object: str
    requested_action: str
    attributes: dict
    provenance: dict


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.strip()


def _derive_issue(content: str) -> str:
    """Neutral restatement: strip admin/identity boilerplate, keep the complaint.

    A pure, auditable transform -- not a paraphrase.
    """
    text = _clean_text(content)
    text = _ADMIN_CLAUSE.sub("", text)
    # Drop a lone leading "Скарга на ..." wrapper when the rest is a request.
    text = re.sub(r"^Щодо\s+", "", text)
    text = re.sub(r"^[Сс]карга\s+на\s+", "", text)
    text = _redact_pii(text)
    return _clean_text(text)[:400]


def _derive_object(content: str, record: dict) -> str:
    """Target of the complaint derived from the text itself.

    Uses the street/address phrase *mentioned in the text*; falls back to an
    explicit address in the free text. Never reads the record's metadata
    columns (those are rarely present in the text and cannot be extracted).
    """
    text = _clean_text(content)
    m = _STREET_PHRASE.search(text)
    if m:
        return f"{m.group('stem')} {m.group('name')}".strip()
    return ""


def _derive_requested_action(content: str) -> str:
    """Sentence containing a request verb, else empty. Never invented."""
    text = _clean_text(content)
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if _REQ_VERBS.search(sent):
            sent = _ADMIN_CLAUSE.sub("", sent)
            return _redact_pii(_clean_text(sent))[:200]
    return ""


def _derive_attributes(content: str, record: dict) -> dict:
    """Small bag of *facts present in the text* (schema says attributes are
    additional structured facts present in the text).

    Only text-derived facts are emitted; the record's metadata columns are not
    in the complaint text, so they are deliberately excluded (no hallucination).
    """
    attrs: dict = {}
    text = _clean_text(content)
    m = _STREET_PHRASE.search(text)
    if m:
        attrs["street"] = f"{m.group('stem')} {m.group('name')}".strip()
    bm = _BUILDING_NO.search(text)
    if bm:
        attrs["building"] = bm.group(1)
    org = record.get("organizationName")
    if org and org not in ("null", "None", "") and org.lower() in text.lower():
        attrs["organization"] = str(org)
    return attrs


def label_record(record: dict) -> LabeledRecord:
    """Deterministic weak labeling of one raw record."""
    content = record.get("content") or ""
    kind = record.get("kind") or ""
    domain = KIND_TO_DOMAIN.get(kind, "other")
    return LabeledRecord(
        uid=record.get("uid", ""),
        domain=domain,
        issue=_derive_issue(content),
        object=_derive_object(content, record),
        requested_action=_derive_requested_action(content),
        attributes=_derive_attributes(content, record),
        provenance={"kind": kind, "source": "weak-deterministic"},
    )


def to_json(labeled: LabeledRecord) -> dict:
    return {
        "topics": [
            {
                "domain": labeled.domain,
                "issue": labeled.issue,
                "object": labeled.object,
                "requested_action": labeled.requested_action,
                "attributes": labeled.attributes,
            }
        ]
    }


SYSTEM_PROMPT = """Ти — система, що перетворює звернення громадян до органів місцевого самоврядування у структурований JSON-формат.

Виведи виключно валідний JSON без пояснень, без markdown-рамок, без коментарів. Схема:
{"topics": [{"domain": str, "issue": str, "object": str, "requested_action": str, "attributes": {}}]}

Правила:
- domain — одна зі значень: roads, water, heating, housing, transport, sanitation, electricity, construction, benefits, government, commerce, payments, other.
- issue — конкретна проблема, сформульована нейтрально, без скарг на конкретних осіб.
- object — адреса або установа, якої стосується звернення. Якщо немає — порожній рядок.
- requested_action — що заявник просить зробити. Якщо прохання немає — порожній рядок.
- attributes — додаткові факти (організація, вулиця, будинок, статус, дата). Лише те, що реально є у тексті.
- topics може містити декілька незалежних проблем. Якщо звернення не містить дієвої проблеми — topics порожній масив.
- НЕ вигадуй факти. Якщо чогось немає у тексті — не додавай. Не включай персональних даних (ПІБ, номери телефонів)."""


def build_chat_example(record: dict) -> dict:
    labeled = label_record(record)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (record.get("content") or "").strip()},
            {"role": "assistant", "content": json.dumps(to_json(labeled), ensure_ascii=False)},
        ],
        "uid": labeled.uid,
    }


def load_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def build(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "validation", "test"):
        rows = load_split(split)
        examples = [build_chat_example(r) for r in rows]
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        summary[split] = len(examples)
        print(f"{split}: {len(examples)} -> {out_path}")
    # mlx_lm load_local_dataset requires a split literally named "valid".
    alias = out_dir / "valid.jsonl"
    if not alias.exists():
        alias.symlink_to(out_dir / "validation.jsonl")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "built_by": "ml/tune/build_dataset.py",
                "schema": "ml/data/gold/annotation_schema.json",
                "system_prompt": SYSTEM_PROMPT,
                "summary": summary,
                "note": "Weak deterministic labels. Not gold. See ml/data/design_mlx_finetune.md.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    build()