import { expect, test } from '@playwright/test';

/**
 * Purpose: Protects the complete navigation and default optimization journey.
 * Description: Opens the dashboard, visits every primary route, runs the sample project, and verifies rendered sheet geometry.
 */
test('navigates the app and optimizes the sample project', async ({ page }) => {
  await page.goto('/');

  // Expected outcome: The product dashboard loads at the application root.
  // Acceptance criteria: The level-one heading describes the application's sheet-saving objective.
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Make every sheet count.');

  for (const route of ['Parts', 'Sheets', 'Settings', 'Optimize']) {
    await page.getByRole('link', { name: route, exact: true }).click();

    // Expected outcome: Every primary navigation link resolves to a meaningful page.
    // Acceptance criteria: Each destination exposes a level-one heading matching its navigation label or expanded stock label.
    const expectedHeading = route === 'Sheets' ? 'Plywood sheets' : route === 'Settings' ? 'Cutting settings' : route === 'Optimize' ? 'Optimizer' : route;
    await expect(page.getByRole('heading', { level: 1 })).toContainText(expectedHeading);
  }

  await page.getByTestId('run-optimizer').click();

  // Expected outcome: The sample project produces at least one inspectable sheet layout.
  // Acceptance criteria: The first SVG sheet and at least one stable placed-part element are visible.
  await expect(page.getByTestId('sheet-layout-0')).toBeVisible();
  await expect(page.locator('[data-testid^="placed-part-"]').first()).toBeVisible();

  // Expected outcome: The known fitting sample leaves no parts unplaced.
  // Acceptance criteria: The unplaced-parts alert is absent after optimization.
  await expect(page.getByTestId('unplaced-parts')).toHaveCount(0);
});
