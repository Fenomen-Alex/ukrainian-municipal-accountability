import { test, expect, type Page, type Route } from '@playwright/test';

const SAMPLE = {
  structured: {
    topics: [
      {
        domain: 'roads',
        issue: 'Не горять ліхтарі',
        object: 'вул. Шевченка, 24',
        requested_action: 'Відновити освітлення',
        attributes: { вулиця: 'Шевченка' },
      },
    ],
  },
};

async function mockAnalyze(page: Page, handler: (route: Route) => Promise<unknown> | unknown) {
  await page.route('**/api/analyze', handler);
}

test('empty submit shows validation error without network call', async ({ page }) => {
  let requests = 0;
  await mockAnalyze(page, async (route) => {
    requests++;
    return route.fulfill({ json: SAMPLE });
  });
  await page.goto('/app');
  await page.getByRole('button', { name: 'Допомогти оформити' }).click();
  await expect(page.locator('#error1')).toBeVisible();
  await expect(page.locator('#error1')).toContainText('опишіть проблему');
  expect(requests).toBe(0);
});

test('character counter updates and blocks over-limit', async ({ page }) => {
  await page.goto('/app');
  await page.locator('#complaintText').fill('а'.repeat(10001));
  await expect(page.locator('#counter')).toContainText('10001 / 10000');
  await expect(page.locator('#counter')).toHaveClass(/counter--over/);
  await page.getByRole('button', { name: 'Допомогти оформити' }).click();
  await expect(page.locator('#error1')).toContainText('занадто довгий');
});

test('happy path: analyze → structure → review → appeal', async ({ page }) => {
  await mockAnalyze(page, (route) => route.fulfill({ json: SAMPLE }));
  await page.goto('/app');
  await page.locator('#complaintText').fill('На вулиці Шевченка не горять ліхтарі');
  await page.getByRole('button', { name: 'Допомогти оформити' }).click();

  await expect(page.locator('#step2')).toBeVisible();
  await expect(page.locator('#step1')).toBeHidden();
  await expect(page.locator('.stepper-item[data-step="2"]')).toHaveClass(/is-active/);
  await expect(page.locator('#topicsContainer')).toContainText('Дороги');
  await expect(page.locator('#topicsContainer textarea').first()).toHaveValue(
    'Не горять ліхтарі',
  );

  await page.locator('#toReviewBtn').click();
  await expect(page.locator('#step3')).toBeVisible();
  await expect(page.locator('#reviewContainer')).toContainText('Не горять ліхтарі');
  await expect(page.locator('#reviewContainer')).toContainText('Дороги');
  await expect(page.locator('#reviewContainer')).toContainText('вул. Шевченка, 24');

  await page.locator('#generateAppealBtn').click();
  await expect(page.locator('#step4')).toBeVisible();
  await expect(page.locator('#appealText')).toHaveValue(/Звернення громадянина/);
  await expect(page.locator('#appealText')).toHaveValue(/вул. Шевченка, 24/);
  await expect(page.locator('#appealText')).toHaveValue(/Не горять ліхтарі/);
  await expect(page.locator('#statusMsg')).toBeVisible();
});

test('draft is saved to localStorage with status ready', async ({ page }) => {
  await mockAnalyze(page, (route) => route.fulfill({ json: SAMPLE }));
  await page.goto('/app');
  await page.locator('#complaintText').fill('На вулиці Шевченка не горять ліхтарі');
  await page.getByRole('button', { name: 'Допомогти оформити' }).click();
  await page.locator('#toReviewBtn').click();
  await page.locator('#generateAppealBtn').click();

  const drafts = await page.evaluate(() =>
    JSON.parse(localStorage.getItem('uma_drafts_v1') || '[]'),
  );
  expect(drafts.length).toBe(1);
  expect(drafts[0].status).toBe('ready');
  expect(drafts[0].originalText).toContain('Шевченка');
});

test('model output HTML is rendered as text, not executed', async ({ page }) => {
  const XSS = {
    structured: {
      topics: [
        {
          domain: 'other',
          issue: '<img src=x onerror="window.__xss=1">',
          object: '<script>window.__xss2=1</script>',
          requested_action: '',
          attributes: {},
        },
      ],
    },
  };
  await mockAnalyze(page, (route) => route.fulfill({ json: XSS }));
  const dialogs: string[] = [];
  page.on('dialog', (d) => {
    dialogs.push(d.message());
    d.dismiss();
  });
  await page.goto('/app');
  await page.locator('#complaintText').fill('тест');
  await page.getByRole('button', { name: 'Допомогти оформити' }).click();
  await expect(page.locator('#step2')).toBeVisible();
  await expect(page.locator('#topicsContainer textarea').first()).toHaveValue(
    '<img src=x onerror="window.__xss=1">',
  );
  await page.locator('#toReviewBtn').click();
  await expect(page.locator('#reviewContainer')).toContainText('<img src=x');
  expect(await page.locator('#reviewContainer img').count()).toBe(0);
  expect(await page.evaluate(() => (window as unknown as { __xss?: number }).__xss)).toBeUndefined();
  expect(dialogs).toEqual([]);
});

test('server validation error (400) shows detail message', async ({ page }) => {
  await mockAnalyze(page, (route) =>
    route.fulfill({
      status: 400,
      json: {
        error: 'Помилка валідації',
        details: [{ path: 'text', message: 'Текст занадто довгий (максимум 10000 символів)' }],
      },
    }),
  );
  await page.goto('/app');
  await page.locator('#complaintText').fill('тест');
  await page.getByRole('button', { name: 'Допомогти оформити' }).click();
  await expect(page.locator('#error1')).toContainText('занадто довгий');
});

test('network failure shows friendly error and re-enables button', async ({ page }) => {
  await mockAnalyze(page, (route) => route.abort('failed'));
  await page.goto('/app');
  await page.locator('#complaintText').fill('тест');
  const btn = page.getByRole('button', { name: 'Допомогти оформити' });
  await btn.click();
  await expect(page.locator('#error1')).toBeVisible();
  await expect(page.locator('#error1')).toContainText('з’єднання');
  await expect(btn).toBeEnabled();
});

test('429 rate limit shows user-facing message', async ({ page }) => {
  await mockAnalyze(page, (route) =>
    route.fulfill({
      status: 429,
      json: { error: 'Забагато запитів. Зачекайте хвилину та спробуйте ще раз.' },
    }),
  );
  await page.goto('/app');
  await page.locator('#complaintText').fill('тест');
  await page.getByRole('button', { name: 'Допомогти оформити' }).click();
  await expect(page.locator('#error1')).toContainText('Забагато запитів');
});

test('step2 blocks review when all topics are empty', async ({ page }) => {
  const EMPTY = {
    structured: { topics: [{ domain: 'other', issue: '', object: '', requested_action: '', attributes: {} }] },
  };
  await mockAnalyze(page, (route) => route.fulfill({ json: EMPTY }));
  await page.goto('/app');
  await page.locator('#complaintText').fill('тест');
  await page.getByRole('button', { name: 'Допомогти оформити' }).click();
  await expect(page.locator('#step2')).toBeVisible();
  await page.locator('#toReviewBtn').click();
  await expect(page.locator('#error2')).toBeVisible();
  await expect(page.locator('#error2')).toContainText('Заповніть хоча б опис проблеми');
  await expect(page.locator('#step3')).toBeHidden();
});

test('real /api/analyze returns structured topics', async ({ request }) => {
  test.skip(!process.env.PLAYWRIGHT_BASE_URL, 'production-only: real inference round-trip');
  const res = await request.post('/api/analyze', {
    data: { text: 'На вулиці Шевченка біля будинку 24 уже тиждень не горять ліхтарі' },
    timeout: 90000,
  });
  expect(res.status()).toBe(200);
  const body = await res.json();
  expect(Array.isArray(body.structured.topics)).toBe(true);
  expect(body.structured.topics.length).toBeGreaterThan(0);
  expect(body).not.toHaveProperty('raw');
});
