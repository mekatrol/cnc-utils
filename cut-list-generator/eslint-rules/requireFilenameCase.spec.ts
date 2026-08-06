import type { ESLint } from 'eslint';
import { Linter } from 'eslint';
import { describe, expect, it } from 'vitest';

import requireFilenameCase from './requireFilenameCase.js';

const lintFilename = (filename: string): Linter.LintMessage[] => {
  const linter = new Linter();
  const localPlugin = {
    rules: {
      requireFilenameCase
    }
  } as unknown as ESLint.Plugin;

  return linter.verify(
    'const example = true;',
    {
      files: ['**/*.ts', '**/*.vue'],
      plugins: {
        local: localPlugin
      },
      rules: {
        'local/requireFilenameCase': 'error'
      }
    },
    { filename }
  );
};

describe('requireFilenameCase', () => {

  /**
   * Purpose: Protects the behavioral contract that requires component implementation files to use PascalCase.
   * Description: Exercises requires component implementation files to use PascalCase from its arranged starting state and
   * verifies the observable results required by the scenario.
   */
  it('requires component implementation files to use PascalCase', () => {

    // Expected outcome: `lintFilename('src/components/appThemeSelector.ts')` matches the required structure.
    // Acceptance criteria: `lintFilename('src/components/appThemeSelector.ts')` must equal `[ expect.objectContaining({ messageId: 'expectedPascalCase' }`, because this condition proves that
    // requires component implementation files to use PascalCase.
    expect(lintFilename('src/components/appThemeSelector.ts')).toEqual([
      expect.objectContaining({
        messageId: 'expectedPascalCase'
      })
    ]);

    // Expected outcome: `lintFilename('src/components/AppThemeSelector.ts')` matches the required structure.
    // Acceptance criteria: `lintFilename('src/components/AppThemeSelector.ts')` must equal `[]`, because this condition proves that
    // requires component implementation files to use PascalCase.
    expect(lintFilename('src/components/AppThemeSelector.ts')).toEqual([]);
  });

  /**
   * Purpose: Protects the behavioral contract that requires Vue component filenames to start with App.
   * Description: Exercises requires Vue component filenames to start with App from its arranged starting state and
   * verifies the observable results required by the scenario.
   */
  it('requires Vue component filenames to start with App', () => {

    // Expected outcome: `lintFilename('src/components/ThemeSelector.vue')` matches the required structure.
    // Acceptance criteria: `lintFilename('src/components/ThemeSelector.vue')` must equal `[ expect.objectContaining({ messageId: 'appPrefixRequired' }`, because this condition proves that
    // requires Vue component filenames to start with App.
    expect(lintFilename('src/components/ThemeSelector.vue')).toEqual([
      expect.objectContaining({
        messageId: 'appPrefixRequired'
      })
    ]);

    // Expected outcome: `lintFilename('src/components/AppThemeSelector.vue')` matches the required structure.
    // Acceptance criteria: `lintFilename('src/components/AppThemeSelector.vue')` must equal `[]`, because this condition proves that
    // requires Vue component filenames to start with App.
    expect(lintFilename('src/components/AppThemeSelector.vue')).toEqual([]);
  });

  /**
   * Purpose: Protects the behavioral contract that requires component test files to use camelCase.
   * Description: Exercises requires component test files to use camelCase from its arranged starting state and
   * verifies the observable results required by the scenario.
   */
  it('requires component test files to use camelCase', () => {

    // Expected outcome: `lintFilename('src/components/__tests__/ThemeSelector.spec.ts')` matches the required structure.
    // Acceptance criteria: `lintFilename('src/components/__tests__/ThemeSelector.spec.ts')` must equal `[ expect.objectContaining({ messageId: 'expectedCamelCase' }`, because this condition proves that
    // requires component test files to use camelCase.
    expect(lintFilename('src/components/__tests__/ThemeSelector.spec.ts')).toEqual([
      expect.objectContaining({
        messageId: 'expectedCamelCase'
      })
    ]);

    // Expected outcome: `lintFilename('src/components/__tests__/themeSelector.spec.ts')` matches the required structure.
    // Acceptance criteria: `lintFilename('src/components/__tests__/themeSelector.spec.ts')` must equal `[]`, because this condition proves that
    // requires component test files to use camelCase.
    expect(lintFilename('src/components/__tests__/themeSelector.spec.ts')).toEqual([]);
  });
});
