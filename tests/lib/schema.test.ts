import { describe, expect, it } from "vitest";
import { ComplaintInput, StructuredComplaint } from "../../src/lib/schema.js";

describe("ComplaintInput", () => {
  it("rejects empty", () => {
    expect(ComplaintInput.safeParse({ text: "" }).success).toBe(false);
  });
  it("enforces max length", () => {
    const long = "a".repeat(10001);
    expect(ComplaintInput.safeParse({ text: long }).success).toBe(false);
  });
});

describe("StructuredComplaint", () => {
  it("accepts empty topics", () => {
    expect(StructuredComplaint.safeParse({ topics: [] }).success).toBe(true);
  });
  it("accepts valid topic", () => {
    expect(StructuredComplaint.safeParse({
      topics: [{
        domain: "roads",
        issue: "яма",
        object: "",
        requested_action: "",
        attributes: {},
      }],
    }).success).toBe(true);
  });
});
