import { test, expect, type Page } from '@playwright/test';

const SAMPLE = {
  structured: {
    topics: [
      {
        domain: 'electricity',
        issue: 'Не працює вуличне освітлення біля школи',
        object: 'вул. Грушевського, 12',
        requested_action: 'відновити освітлення',
        attributes: { street: 'Грушевського' },
      },
      {
        domain: 'roads',
        issue: 'Яма на тротуарі біля зупинки',
        object: '',
        requested_action: '',
        attributes: {},
      },
    ],
  },
};

async function runToAppeal(page: Page) {
  await page.route('**/api/analyze', (route) => route.fulfill({ json: SAMPLE }));
  await page.goto('/app');
  await page.locator('#complaintText').fill('На вулиці Грушевського не горять ліхтарі та яма на тротуарі');
  await page.getByRole('button', { name: 'Допомогти оформити' }).click();
  await expect(page.locator('#step2')).toBeVisible();
  await page.locator('#toReviewBtn').click();
  await expect(page.locator('#step3')).toBeVisible();
  await page.locator('#generateAppealBtn').click();
  await expect(page.locator('#step4')).toBeVisible();
  await expect(page.locator('#appealText')).not.toHaveValue('');
  return page.locator('#appealText').inputValue();
}

test('appeal document uses Ukrainian domain labels, not raw enums', async ({ page }) => {
  const appeal = await runToAppeal(page);
  expect(appeal).toContain('Сфера: Електропостачання');
  expect(appeal).toContain('Сфера: Дороги');
  expect(appeal).not.toContain('Сфера: electricity');
  expect(appeal).not.toContain('Сфера: roads');
});

test('appeal document has addressee and signature placeholders', async ({ page }) => {
  const appeal = await runToAppeal(page);
  expect(appeal).toMatch(/До[: ].*орган місцевого самоврядування/s);
  expect(appeal).toMatch(/Дата[: ]/);
  expect(appeal).toMatch(/Підпис[: ]/);
});

test('appeal document puts channel and disclaimer in a footer after the signature', async ({ page }) => {
  const appeal = await runToAppeal(page);
  const sigIdx = appeal.indexOf('Підпис');
  const channelIdx = appeal.indexOf('Канал надсилання');
  const disclaimerIdx = appeal.indexOf('Звернення підготовлене за допомогою асистента');
  expect(sigIdx).toBeGreaterThan(-1);
  expect(channelIdx).toBeGreaterThan(sigIdx);
  expect(disclaimerIdx).toBeGreaterThan(channelIdx);
});

test('appeal document labels attribute keys in Ukrainian', async ({ page }) => {
  const appeal = await runToAppeal(page);
  expect(appeal).toContain('- Вулиця: Грушевського');
  expect(appeal).not.toContain('- street:');
});
