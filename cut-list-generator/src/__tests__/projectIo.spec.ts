import { describe, expect, it } from 'vitest';
import { parseProject } from '@/utils/projectIo';

describe('project import', () => {
  /**
   * Purpose: Ensures malformed imports cannot overwrite current project data.
   * Description: Parses syntactically valid JSON that lacks the required versioned project structure.
   */
  it('rejects invalid project json', () => {
    // Expected outcome: Invalid project structure fails before any store mutation can occur.
    // Acceptance criteria: Parsing throws an explicit validation error.
    expect(() => parseProject('{"parts":[]}')).toThrow('Unsupported or incomplete project file.');
  });

  /**
   * Purpose: Keeps older version-one project exports compatible with production sets.
   * Description: Imports a valid project that predates the set-count field.
   */
  it('defaults legacy projects to one production set', () => {
    const project = parseProject(JSON.stringify({ version: 1, parts: [], sheets: [], settings: { kerf: 3, edgeMargin: 10, allowRotation: true, strategy: 'best-area-fit', maxIterations: 20 } }));

    // Expected outcome: A legacy project represents one copy of its cut list.
    // Acceptance criteria: The normalized project has a set count of one.
    expect(project.setCount).toBe(1);
  });
});
