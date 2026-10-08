import { test, expect } from '@playwright/test';

const REQUEST_TEXT =
  'Уже третій тиждень на вулиці Грушевського, 12 біля школи №4 не працює вуличне освітлення — темно після 20:00. ' +
  'Крім того, на тротуарі біля зупинки утворилася яма. ' +
  'Прошу відновити освітлення та відремонтувати тротуар.';

test('real model: fills requested_action for explicit requests', async ({ request }) => {
  test.skip(!process.env.PLAYWRIGHT_BASE_URL, 'production-only: real inference round-trip');
  const res = await request.post('/api/analyze', {
    data: { text: REQUEST_TEXT },
    timeout: 120000,
  });
  expect(res.status()).toBe(200);
  const body = await res.json();
  const topics = body.structured.topics;
  expect(Array.isArray(topics)).toBe(true);
  expect(topics.length).toBeGreaterThan(0);
  const withAction = topics.filter(
    (t: { requested_action: string }) => t.requested_action.trim().length > 0,
  );
  expect(withAction.length).toBeGreaterThan(0);
});

test('real model: issue text contains no narration tails or request sentences', async ({ request }) => {
  test.skip(!process.env.PLAYWRIGHT_BASE_URL, 'production-only: real inference round-trip');
  const res = await request.post('/api/analyze', {
    data: { text: REQUEST_TEXT },
    timeout: 120000,
  });
  expect(res.status()).toBe(200);
  const body = await res.json();
  const blob = JSON.stringify(body.structured.topics);
  expect(blob).not.toMatch(/заявниц[іяи]|заявник|громадян(?:ин|ка)/i);
  for (const t of body.structured.topics) {
    expect(t.issue).not.toMatch(/(^|\.\s)Прошу/i);
    expect(t.requested_action).not.toMatch(/заявниц[іяи]|заявник/i);
  }
});
