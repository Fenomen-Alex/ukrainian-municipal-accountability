import { test, expect } from '@playwright/test';

test('landing page loads', async ({ page }) => {
  await page.goto('http://localhost:3000/');
  await expect(page).toHaveTitle(/Кропивницький/);
  await expect(page.getByText('Описати проблему')).toBeVisible();
});

test('app has input form', async ({ page }) => {
  await page.goto('http://localhost:3000/app');
  await expect(page.getByLabel(/Опис проблеми/)).toBeVisible();
});
