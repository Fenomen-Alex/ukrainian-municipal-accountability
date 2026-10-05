import { describe, expect, it, beforeEach } from "vitest";
import { loadDrafts, saveDraft, getDraft, newDraftId } from "../../src/services/drafts.js";

describe("drafts", () => {
  beforeEach(() => {
    try {
      localStorage.clear();
    } catch (e) {
      // ignore
    }
  });

  it("creates and saves draft", () => {
    const id = newDraftId();
    const draft = {
      id,
      createdAt: new Date().toISOString(),
      originalText: "тест",
      structured: { topics: [] },
      finalAppeal: "текст",
      channelUrl: "https://example.ua/",
      status: "ready" as const,
    };
    saveDraft(draft);
    const loaded = loadDrafts();
    expect(loaded.length).toBe(1);
    expect(getDraft(id)?.status).toBe("ready");
  });

  it("updates existing draft", () => {
    const id = newDraftId();
    const draft = {
      id,
      createdAt: new Date().toISOString(),
      originalText: "тест",
      structured: { topics: [] },
      finalAppeal: "текст",
      channelUrl: "https://example.ua/",
      status: "ready" as const,
    };
    saveDraft(draft);
    saveDraft({ ...draft, status: "submitted_to_user" });
    expect(loadDrafts().length).toBe(1);
    expect(getDraft(id)?.status).toBe("submitted_to_user");
  });

  it("preserves original complaint", () => {
    const id = newDraftId();
    const orig = "У дворі не працює освітлення";
    saveDraft({
      id,
      createdAt: new Date().toISOString(),
      originalText: orig,
      structured: { topics: [] },
      finalAppeal: "appeal",
      channelUrl: "https://example.ua/",
      status: "ready",
    });
    expect(getDraft(id)?.originalText).toBe(orig);
  });
});
