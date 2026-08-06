import process from 'node:process';

import js from '@eslint/js';
import type { ESLint, Linter } from 'eslint';
import { defineConfig, globalIgnores } from 'eslint/config';
import globals from 'globals';
import vue from 'eslint-plugin-vue';
import tseslint from 'typescript-eslint';
import vueParser from 'vue-eslint-parser';

import requireAutomationProp from './eslint-rules/requireAutomationProp.js';
import requireFilenameCase from './eslint-rules/requireFilenameCase.js';

const localPlugin = {
  rules: {
    requireAutomationProp,
    requireFilenameCase
  }
} as unknown as ESLint.Plugin;

const sharedRules: Linter.RulesRecord = {
  'local/requireFilenameCase': 'error',

  'prefer-promise-reject-errors': 'error',

  'max-len': [
    'error',
    200,
    {
      ignorePattern: '^\\s*\\/'
    }
  ],

  quotes: [
    'error',
    'single',
    {
      avoidEscape: true,
      allowTemplateLiterals: false
    }
  ],

  '@typescript-eslint/no-var-requires': 'off',

  'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'off',

  'no-console': [
    process.env.NODE_ENV === 'production' ? 'error' : 'warn',
    {
      allow: ['warn', 'error']
    }
  ],

  'no-unused-vars': 'off',
  'no-var': 'error',

  'no-restricted-imports': [
    'error',
    {
      patterns: [
        {
          group: ['../**'],
          message:
            "Alias import '@' must be used for imports from outside the current directory. Same-directory './' imports are allowed."
        }
      ]
    }
  ],

  'prefer-arrow-callback': 'error',

  'func-style': [
    'error',
    'expression',
    {
      allowArrowFunctions: true
    }
  ],

  '@typescript-eslint/no-unused-vars': [
    'error',
    {
      argsIgnorePattern: '^_.*$',
      varsIgnorePattern: '^_$',
      caughtErrorsIgnorePattern: '^_$'
    }
  ],

  '@typescript-eslint/explicit-function-return-type': [
    'error',
    {
      allowExpressions: true,
      allowHigherOrderFunctions: true,
      allowTypedFunctionExpressions: true
    }
  ]
};

const vueRules: Linter.RulesRecord = {
  ...sharedRules,

  'local/requireAutomationProp': 'error',

  'vue/html-closing-bracket-newline': 'off',
  'vue/html-self-closing': 'off',
  'vue/max-attributes-per-line': 'off',
  'vue/singleline-html-element-content-newline': 'off',

  'vue/block-order': [
    'error',
    {
      order: ['template', 'script', 'style']
    }
  ]
};

export default defineConfig(
  globalIgnores(['dist/**', 'harness/dist/**', 'node_modules/**']),

  js.configs.recommended,

  {
    languageOptions: {
      globals: globals.browser
    }
  },

  tseslint.configs.recommended,
  vue.configs['flat/recommended'],

  {
    plugins: {
      local: localPlugin
    }
  },

  {
    files: ['**/*.ts'],

    languageOptions: {
      parser: tseslint.parser,

      parserOptions: {
        projectService: true
      }
    },

    rules: sharedRules
  },

  {
    files: ['**/*.vue'],

    languageOptions: {
      parser: vueParser,

      parserOptions: {
        parser: tseslint.parser,
        projectService: true,
        extraFileExtensions: ['.vue']
      }
    },

    rules: vueRules
  }
);
