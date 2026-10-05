import { describe, expect, it } from "vitest";
import { generateAppeal } from "../../src/services/appeal.js";

describe("hallucination safety", () => {
  it("does not invent facts", () => {
    const structured = {
      topics: [{
        domain: "roads" as const,
        issue: "яма",
        object: "",
        requested_action: "",
        attributes: {},
      }],
    };
    const appeal = generateAppeal(structured, { channelUrl: "https://example.ua/" });
    expect(appeal).not.toMatch(/\d{2}\.\d{2}\.\d{4}/);
    expect(appeal).not.toMatch(/департамент|управління|інспекція/);
    expect(appeal).not.toMatch(/терміново|невідкладно/);
  });

  it("preserves empty requested action correctly", () => {
    const structured = {
      topics: [{
        domain: "other" as const,
        issue: "проблема",
        object: "",
        requested_action: "",
        attributes: {},
      }],
    };
    const appeal = generateAppeal(structured, { channelUrl: "https://example.ua/" });
    expect(appeal).toContain("Проблема: проблема");
    expect(appeal).not.toMatch(/Прошу:\s*$/m);
    expect(appeal).not.toContain("requested_action");
  });
});
