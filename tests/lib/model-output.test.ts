import { describe, expect, it } from "vitest";
import { parseModelContent } from "../../src/services/inference.js";

describe("parseModelContent", () => {
  it("parses a bare topic object followed by degenerate '!' spam (observed in production)", () => {
    const content =
      '\n{"domain": "electricity", "issue": "не горять ліхтарі", "object": "вул. Шевченка", "requested_action": "відновити", "attributes": {"street": "Шевченка"}}\n!\n!\n!\n!\n';
    const out = parseModelContent(content);
    expect(out.topics).toHaveLength(1);
    expect(out.topics[0]!.domain).toBe("electricity");
    expect(out.topics[0]!.issue).toBe("не горять ліхтарі");
  });

  it("parses a JSON array of topics", () => {
    const content = JSON.stringify([
      { domain: "roads", issue: "яма", object: "вул. Перемоги", requested_action: "", attributes: {} },
      { domain: "water", issue: "витік", object: "", requested_action: "ремонт", attributes: {} },
    ]);
    const out = parseModelContent(content);
    expect(out.topics).toHaveLength(2);
  });

  it("parses a {topics: [...]} object", () => {
    const content = JSON.stringify({
      topics: [{ domain: "other", issue: "x", object: "y", requested_action: "", attributes: {} }],
    });
    const out = parseModelContent(content);
    expect(out.topics).toHaveLength(1);
  });

  it("ignores markdown fences and surrounding text", () => {
    const content =
      'Ось результат:\n```json\n[{"domain": "roads", "issue": "яма", "object": "вул. Шевченка", "requested_action": "", "attributes": {}}]\n```\nГотово.';
    const out = parseModelContent(content);
    expect(out.topics).toHaveLength(1);
    expect(out.topics[0]!.domain).toBe("roads");
  });

  it("handles escaped quotes and braces inside strings", () => {
    const content =
      '{"domain": "other", "issue": "заява \\"закрити\\" {дужки} [квадрати]", "object": "", "requested_action": "", "attributes": {"note": "a]b}c"}}\n!\n';
    const out = parseModelContent(content);
    expect(out.topics[0]!.issue).toContain('закрити');
  });

  it("parses comma-separated bare objects without an enclosing array (observed in production)", () => {
    const content =
      '\n{"domain": "roads", "issue": "ями на Соборній", "object": "Соборна", "requested_action": "", "attributes": {}}, {"domain": "sanitation", "issue": "сміття біля школи", "object": "", "requested_action": "", "attributes": {}}\n!\n!\n';
    const out = parseModelContent(content);
    expect(out.topics).toHaveLength(2);
    expect(out.topics[0]!.domain).toBe("roads");
    expect(out.topics[1]!.domain).toBe("sanitation");
  });

  it("throws on content without JSON", () => {
    expect(() => parseModelContent("немає тут нічого ! ! !")).toThrow();
  });

  it("throws on truncated JSON", () => {
    expect(() => parseModelContent('{"domain": "roads", "issue": "не за')).toThrow();
  });
});
