import type { ComplaintInput } from "./schema.js";

const DOUBLE_QUOTE = /"([^"]*)"/g;
const SINGLE_QUOTES = /(\p{L})"(\p{L})/gu;

export function normalizeUserText(text: string): string {
  let normalized = text.replace(SINGLE_QUOTES, "$1’$2");
  normalized = normalized.replace(DOUBLE_QUOTE, (match, content) => {
    if (content.length === 0) return match;
    return `“${content}”`;
  });
  return normalized.trim();
}

export function normalizeComplaintInput(input: ComplaintInput): ComplaintInput {
  return {
    text: normalizeUserText(input.text),
    location: input.location?.trim() || undefined,
    contact: input.contact?.trim() || undefined,
  };
}
