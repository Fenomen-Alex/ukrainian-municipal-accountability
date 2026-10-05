import { describe, expect, it } from "vitest";
import { normalizeUserText, normalizeComplaintInput } from "../../src/lib/normalize.js";
import { ComplaintInput } from "../../src/lib/schema.js";

describe("normalizeUserText", () => {
  it("handles empty quotes", () => {
    expect(normalizeUserText('текст "текст"')).toContain("“");
    expect(normalizeUserText('текст "текст"')).toContain("”");
  });

  it("handles mid-word quotes", () => {
    expect(normalizeUserText('під"їзду')).toBe("під’їзду");
    expect(normalizeUserText('роз"яснення')).toBe("роз’яснення");
  });

  it("preserves Ukrainian text", () => {
    const s = "У дворі проблема";
    expect(normalizeUserText(s)).toBe(s);
  });
});

describe("ComplaintInput validation", () => {
  it("rejects empty complaint", () => {
    expect(ComplaintInput.safeParse({ text: "" }).success).toBe(false);
  });
  it("accepts minimal valid", () => {
    expect(ComplaintInput.safeParse({ text: "проблема" }).success).toBe(true);
  });
  it("allows optional fields", () => {
    const r = ComplaintInput.safeParse({ text: "проблема", location: "вул. Шевченка", contact: "test@test.com" });
    expect(r.success).toBe(true);
  });
});
