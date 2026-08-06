export const readStored = <T>(key: string, fallback: T): T => {
  try {
    const value = localStorage.getItem(key);
    return value ? (JSON.parse(value) as T) : structuredClone(fallback);
  } catch {
    return structuredClone(fallback);
  }
};
export const writeStored = (key: string, value: unknown): void => localStorage.setItem(key, JSON.stringify(value));
export const makeId = (prefix: string): string => `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
