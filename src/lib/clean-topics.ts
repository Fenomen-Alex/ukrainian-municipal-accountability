import type { StructuredComplaint, Topic } from "./schema.js";

const NARRATION_TAIL = /\s+(?:заявниц\p{L}*|заявник\p{L}*|громадян\p{L}*)\s*[.!?]*$/iu;
const REQUEST_LEAD = /^(?:Прошу|Просимо|Вимагаю|Вимагаємо)[,\s]+/u;
const REQUEST_SENTENCE = /^(?:Прошу|Просимо|Вимагаю|Вимагаємо)\s+/u;

function stripNarrationTail(text: string): string {
  return text.replace(NARRATION_TAIL, "").trim();
}

function splitSentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+/u).filter((s) => s.length > 0);
}

export function cleanTopic(topic: Topic): Topic {
  let issue = stripNarrationTail(topic.issue);
  let requestedAction = stripNarrationTail(topic.requested_action);

  const kept: string[] = [];
  const extracted: string[] = [];
  for (const sentence of splitSentences(issue)) {
    if (REQUEST_SENTENCE.test(sentence)) extracted.push(sentence);
    else kept.push(sentence);
  }
  if (extracted.length > 0) {
    issue = kept.join(" ").trim();
    if (!requestedAction.trim()) {
      requestedAction = extracted
        .map((s) => s.replace(REQUEST_LEAD, "").trim())
        .join(" ")
        .trim();
    }
  }

  return { ...topic, issue, requested_action: requestedAction };
}

export function cleanTopics(structured: StructuredComplaint): StructuredComplaint {
  return { topics: structured.topics.map(cleanTopic) };
}

export function fillActionsFromRequestText(
  structured: StructuredComplaint,
  sourceText: string,
): StructuredComplaint {
  const topics = structured.topics;
  const emptyIndexes = topics
    .map((t, i) => (t.requested_action.trim() ? -1 : i))
    .filter((i) => i >= 0);
  if (emptyIndexes.length === 0) return structured;

  const parts: string[] = [];
  for (const sentence of splitSentences(sourceText)) {
    if (!REQUEST_SENTENCE.test(sentence)) continue;
    const body = sentence
      .replace(REQUEST_LEAD, "")
      .replace(/[.!?]+$/u, "")
      .trim();
    if (!body) continue;
    const pieces = body
      .split(/\s+та\s+/u)
      .map((p) => p.trim())
      .filter(Boolean);
    if (pieces.length >= 2 && pieces.length <= topics.length - parts.length) {
      parts.push(...pieces);
    } else {
      parts.push(body);
    }
  }
  if (parts.length === 0) return structured;

  const claimed = topics
    .map((t) => t.requested_action.trim())
    .filter(Boolean);
  const unused = parts.filter((p) => !claimed.some((c) => c.includes(p)));
  if (unused.length === 0) return structured;

  const next = topics.map((t) => ({ ...t }));
  let assigned = 0;
  for (const i of emptyIndexes) {
    if (assigned >= unused.length) break;
    next[i]!.requested_action = unused[assigned]!;
    assigned++;
  }
  if (assigned === 0) return structured;
  return { topics: next };
}

export function repairActions(
  structured: StructuredComplaint,
  sourceText: string,
): StructuredComplaint {
  const bodies: string[] = [];
  for (const sentence of splitSentences(sourceText)) {
    if (!REQUEST_SENTENCE.test(sentence)) continue;
    const body = sentence.replace(REQUEST_LEAD, "").replace(/[.!?]+$/u, "").trim();
    if (body) bodies.push(body);
  }
  const topics = structured.topics.map((t) => ({ ...t }));
  let changed = false;

  const lcp = (a: string, b: string): number => {
    const n = Math.min(a.length, b.length);
    let i = 0;
    while (i < n && a[i] === b[i]) i++;
    return i;
  };

  for (const t of topics) {
    const action = t.requested_action.trim();
    if (!action) continue;
    const body = bodies.find((b) => {
      if (b === action) return false;
      const shared = lcp(action, b);
      return shared >= 10 && action.length - shared <= 3;
    });
    if (body) {
      t.requested_action = body;
      changed = true;
    }
  }

  for (const t of topics) {
    const action = t.requested_action.trim();
    if (!action) continue;
    const sentences = splitSentences(t.issue);
    const kept = sentences.filter((s) => s.trim() !== action);
    if (kept.length < sentences.length && kept.length > 0) {
      t.issue = kept.join(" ").trim();
      changed = true;
    }
  }

  return changed ? { topics } : structured;
}
