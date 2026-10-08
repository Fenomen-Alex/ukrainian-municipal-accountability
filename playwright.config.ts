import { defineConfig, devices } from '@playwright/test';

const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:3000';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  retries: process.env.PLAYWRIGHT_BASE_URL ? 1 : 0,
  reporter: 'list',
  use: {
    baseURL,
    trace: 'off',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'mobile',
      use: { ...devices['iPhone 13'] },
    },
  ],
  ...(process.env.PLAYWRIGHT_BASE_URL
    ? {}
    : {
        webServer: {
          command: 'npx tsx src/index.ts',
          url: 'http://localhost:3000',
          reuseExistingServer: true,
          timeout: 120000,
        },
      }),
});
