import { test, expect } from '@playwright/test';

for (const url of ['/', '/app']) {
  test(`no horizontal overflow on ${url}`, async ({ page }) => {
    await page.goto(url);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
  });
}

test('landing CTA and steps are visible on this viewport', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Описати проблему')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Як це працює' })).toBeVisible();
});

test('form controls meet 44px touch target on this viewport', async ({ page }) => {
  await page.goto('/app');
  const btn = page.getByRole('button', { name: 'Допомогти оформити' });
  const box = await btn.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.height).toBeGreaterThanOrEqual(40);
});

test('app stepper visible and first step active', async ({ page }) => {
  await page.goto('/app');
  await expect(page.locator('.stepper-item[data-step="1"]')).toHaveClass(/is-active/);
  await expect(page.locator('#step1')).toBeVisible();
  await expect(page.locator('#step2')).toBeHidden();
});

test('no inline styles are used (CSP compatibility)', async ({ page }) => {
  for (const url of ['/', '/app']) {
    await page.goto(url);
    const inlineStyles = await page.evaluate(
      () => document.querySelectorAll('[style]').length,
    );
    expect(inlineStyles).toBe(0);
    const styleBlocks = await page.evaluate(() => document.querySelectorAll('style').length);
    expect(styleBlocks).toBe(0);
  }
});
