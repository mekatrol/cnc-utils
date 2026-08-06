import { expect, test } from '@playwright/test';

/**
 * Purpose: Protects the user-visible contract that the application root loads successfully.
 * Description: Opens the root URL and verifies that the primary heading contains the startup success message.
 */
test('visits the app root url', async ({ page }) => {
  await page.goto('/');

  // Expected outcome: The application displays its primary startup heading.
  // Acceptance criteria: The level-one heading reads "You did it!" because that is the root page's visible success message.
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('You did it!');
});
