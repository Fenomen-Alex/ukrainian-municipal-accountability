export class SlidingWindowLimiter {
  private hits = new Map<string, number[]>();

  constructor(
    private windowMs: number,
    private max: number,
  ) {}

  check(key: string): { allowed: boolean; retryAfterSec: number } {
    const now = Date.now();
    const cutoff = now - this.windowMs;
    const arr = (this.hits.get(key) || []).filter((t) => t > cutoff);
    if (arr.length >= this.max) {
      const retryAfterSec = Math.max(1, Math.ceil((arr[0]! + this.windowMs - now) / 1000));
      this.hits.set(key, arr);
      return { allowed: false, retryAfterSec };
    }
    arr.push(now);
    this.hits.set(key, arr);
    if (this.hits.size > 10000) {
      for (const [k, v] of this.hits) {
        if (v.every((t) => t <= cutoff)) this.hits.delete(k);
      }
    }
    return { allowed: true, retryAfterSec: 0 };
  }
}
