import type { Draft } from "../lib/schema.js";

const KEY = "uma_drafts_v1";

export function loadDrafts(): Draft[] {
  if (typeof window === "undefined") {
    return [];
  }
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as Draft[]) : [];
  } catch {
    return [];
  }
}

export function saveDraft(draft: Draft): void {
  if (typeof window === "undefined") return;
  const drafts = loadDrafts();
  const existingIndex = drafts.findIndex((d) => d.id === draft.id);
  if (existingIndex >= 0) {
    drafts[existingIndex] = draft;
  } else {
    drafts.unshift(draft);
  }
  try {
    localStorage.setItem(KEY, JSON.stringify(drafts));
  } catch {
    // ignore storage errors
  }
}

export function getDraft(id: string): Draft | undefined {
  return loadDrafts().find((d) => d.id === id);
}

export function newDraftId(): string {
  return "draft_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
}
