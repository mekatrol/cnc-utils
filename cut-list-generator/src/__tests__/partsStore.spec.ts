import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { usePartsStore } from '@/stores/parts';

describe('parts production sets', () => {
  beforeEach(() => {
    const values = new Map<string, string>();
    vi.stubGlobal('localStorage', {
      clear: (): void => values.clear(),
      getItem: (key: string): string | null => values.get(key) ?? null,
      removeItem: (key: string): void => {
        values.delete(key);
      },
      setItem: (key: string, value: string): void => {
        values.set(key, value);
      }
    });
    localStorage.clear();
    setActivePinia(createPinia());
  });

  /**
   * Purpose: Ensures one reusable assembly definition can drive a multi-unit production run.
   * Description: Sets the production count to five and verifies quantities and areas are multiplied without changing each part definition.
   */
  it('multiplies part totals and optimization quantities by the set count', () => {
    const store = usePartsStore();
    store.replace([{ id: 'side', name: 'Side', width: 700, height: 500, quantity: 2, canRotate: true, grainDirection: 'none', material: 'Plywood', thickness: 18 }]);
    store.setCount = 5;

    // Expected outcome: Two sides per cupboard become ten sides for five cupboards.
    // Acceptance criteria: The total and optimizer quantity are ten while the source definition remains two per set.
    expect(store.totalPartCount).toBe(10);
    expect(store.optimizationParts[0]?.quantity).toBe(10);
    expect(store.parts[0]?.quantity).toBe(2);

    // Expected outcome: Area totals reflect every manufactured copy.
    // Acceptance criteria: Ten 700 by 500 millimetre parts total 3,500,000 square millimetres.
    expect(store.totalPartArea).toBe(3_500_000);
  });

  /**
   * Purpose: Protects production set persistence and reset behavior.
   * Description: Changes the set count, creates a fresh store instance, then resets the project defaults.
   */
  it('persists and resets the set count', async () => {
    const store = usePartsStore();
    store.setCount = 5;
    await Promise.resolve();
    setActivePinia(createPinia());
    const restored = usePartsStore();

    // Expected outcome: A reload preserves the requested production run.
    // Acceptance criteria: The newly created store restores five sets from local storage.
    expect(restored.setCount).toBe(5);
    restored.reset();

    // Expected outcome: Resetting sample data also resets production to one assembly.
    // Acceptance criteria: The set count returns to one.
    expect(restored.setCount).toBe(1);
  });
});
