import { defaultSettings, type ProjectData } from '@/domain/models';
import { validatePart, validateSheet } from '@/domain/validation';

export const parseProject = (source: string): ProjectData => {
  const value: unknown = JSON.parse(source);
  if (!value || typeof value !== 'object') throw new Error('Project must be a JSON object.');
  const project = value as Partial<ProjectData>;
  if (project.version !== 1 || !Array.isArray(project.parts) || !Array.isArray(project.sheets) || !project.settings) throw new Error('Unsupported or incomplete project file.');
  if (project.parts.some((part) => Object.keys(validatePart(part)).length) || project.sheets.some((sheet) => Object.keys(validateSheet(sheet)).length))
    throw new Error('Project contains invalid parts or sheets.');
  const settings = { ...defaultSettings, ...project.settings };
  if (settings.kerf < 0 || settings.edgeMargin < 0 || settings.maxIterations < 1) throw new Error('Project contains invalid optimizer settings.');
  const setCount = project.setCount ?? 1;
  if (!Number.isInteger(setCount) || setCount < 1) throw new Error('Project contains an invalid production set count.');
  return { ...(project as Omit<ProjectData, 'setCount'>), settings, setCount };
};
export const downloadJson = (filename: string, value: unknown): void => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};
