# Project code standards

## Frontend lint and formatting

For every source change:

- Follow `eslint.config.ts`, including expression-style functions, explicit TypeScript return types, alias imports across directories, filename casing, and Vue block ordering.
- Use single quotes, semicolons, no trailing commas, two-space indentation, LF line endings, and a maximum line length of 200 characters.
- Run `npm run format` and then `npm run lint` after editing.
- Treat formatter and linter fixes as source changes: inspect the diff, revert unrelated rewrites, and rerun both commands until they pass.
- Run relevant tests and `npm run build` after lint passes. Do not report a change as complete while formatting, linting, tests, type-checking, or the production build fails.

Use `npm run format:check` and `npm run lint:check` when source files must not be modified.

## Test organisation and documentation

- Keep Playwright specs separated by user-facing function; do not accumulate unrelated behavior in a general route spec.
- Each test must verify one independently meaningful behavior and must run independently with fresh mutable state.
- Extract repeated mocks and setup into narrowly scoped helpers or fixtures.
- Prefer role- and label-based Playwright locators. Use direct CSS or SVG selectors only when no accessible locator is exposed.
- Add a documentation block immediately above each `it(...)` or `test(...)` declaration with `Purpose:` and `Description:` lines explaining why the test matters and what scenario it exercises.
- Immediately precede every assertion with `Expected outcome:` and `Acceptance criteria:` comments. State the precise pass condition and why it proves the expected behavior.
- Use Arrange, Act, and Assert comments when they materially improve readability; they do not replace test- or assertion-level documentation.
