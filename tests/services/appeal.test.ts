import { describe, expect, it } from "vitest";
import { generateAppeal } from "../../src/services/appeal.js";

describe("generateAppeal", () => {
  it("generates appeal from structured data", () => {
    const structured = {
      topics: [{
        domain: "electricity" as const,
        issue: "не працює освітлення",
        object: "вул. Шевченка, буд. 1",
        requested_action: "відновити освітлення",
        attributes: { "будинок": "1" },
      }],
    };
    const appeal = generateAppeal(structured, { channelUrl: "https://example.ua/" });
    expect(appeal).toContain("не працює освітлення");
    expect(appeal).toContain("відновити освітлення");
    expect(appeal).toContain("Канал надсилання");
  });

  it("handles empty topics", () => {
    const appeal = generateAppeal({ topics: [] }, { channelUrl: "https://example.ua/" });
    expect(appeal).toContain("Проблема не визначена");
  });

  it("handles multiple topics", () => {
    const structured = {
      topics: [
        { domain: "roads" as const, issue: "яма", object: "", requested_action: "ремонт", attributes: {} },
        { domain: "water" as const, issue: "тече", object: "", requested_action: "усунути", attributes: {} },
      ],
    };
    const appeal = generateAppeal(structured, { channelUrl: "https://example.ua/" });
    expect(appeal).toContain("1.");
    expect(appeal).toContain("2.");
  });

  it("does not add Прошу line if empty", () => {
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
    expect(appeal).not.toMatch(/Прошу:\s*$/m);
  });
});
