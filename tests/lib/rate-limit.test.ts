import { describe, expect, it } from "vitest";
import { SlidingWindowLimiter } from "../../src/lib/rate-limit.js";

describe("SlidingWindowLimiter", () => {
  it("allows up to max then blocks", () => {
    const l = new SlidingWindowLimiter(60000, 3);
    expect(l.check("a").allowed).toBe(true);
    expect(l.check("a").allowed).toBe(true);
    expect(l.check("a").allowed).toBe(true);
    const blocked = l.check("a");
    expect(blocked.allowed).toBe(false);
    expect(blocked.retryAfterSec).toBeGreaterThan(0);
  });

  it("isolates keys", () => {
    const l = new SlidingWindowLimiter(60000, 1);
    expect(l.check("a").allowed).toBe(true);
    expect(l.check("b").allowed).toBe(true);
    expect(l.check("a").allowed).toBe(false);
  });
});
