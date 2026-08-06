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
});
