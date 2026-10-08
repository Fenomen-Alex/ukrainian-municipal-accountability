import { test, expect } from '@playwright/test';

test('landing page loads', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/Кропивницький/);
  await expect(page.getByText('Описати проблему')).toBeVisible();
});

test('app has input form', async ({ page }) => {
  await page.goto('/app');
  await expect(page.getByLabel(/Опис проблеми/)).toBeVisible();
});

test('landing explains how it works and is not official', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Як це працює' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Приватність' })).toBeVisible();
  await expect(page.getByText('Це не офіційний сервіс Кропивницької міської ради')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Ви контролюєте процес' })).toBeVisible();
  await expect(page.getByText('Жодних автоматичних відправок', { exact: false })).toBeVisible();
});

test('landing has no automated submission claims or fake stats', async ({ page }) => {
  await page.goto('/');
  const body = await page.locator('body').innerText();
  expect(body).not.toMatch(/\d+% /); // no percentage statistics
  expect(body).not.toMatch(/гарантовано|100% надійності|рекомендують/);
});

test('security headers are present', async ({ page }) => {
  const resp = await page.goto('/');
  expect(resp).not.toBeNull();
  const headers = resp!.headers();
  expect(headers['content-security-policy']).toContain("default-src 'self'");
  expect(headers['content-security-policy']).toContain("frame-ancestors 'none'");
  expect(headers['x-content-type-options']).toBe('nosniff');
  expect(headers['referrer-policy']).toBe('strict-origin-when-cross-origin');
});
