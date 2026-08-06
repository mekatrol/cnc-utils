import { describe, expect, it } from 'vitest';
import type { OptimizerSettings, PartDefinition, StockSheetDefinition } from '@/domain/models';
import { area, canRotate, intersects, pruneFreeRectangles, splitFreeRectangle } from '@/services/optimizer/geometry';
import { compareResults, expandParts, optimize } from '@/services/optimizer/optimizer';

const settings: OptimizerSettings = { kerf: 3, edgeMargin: 10, allowRotation: true, strategy: 'best-area-fit', maxIterations: 20 };
const part = (overrides: Partial<PartDefinition> = {}): PartDefinition => ({
  id: 'part',
  name: 'Part',
  width: 100,
  height: 50,
  quantity: 1,
  canRotate: true,
  grainDirection: 'none',
  material: 'Plywood',
  thickness: 18,
  ...overrides
});
const sheet = (overrides: Partial<StockSheetDefinition> = {}): StockSheetDefinition => ({
  id: 'sheet',
  name: 'Sheet',
  width: 300,
  height: 200,
  quantity: 1,
  material: 'Plywood',
  thickness: 18,
  ...overrides
});

describe('optimizer domain', () => {
  /**
   * Purpose: Ensures part quantities become uniquely addressable cutting instances.
   * Description: Expands three copies and checks their stable identifiers and total area.
   */
  it('expands quantities and calculates area', () => {
    const instances = expandParts([part({ quantity: 3 })]);
    // Expected outcome: Every requested copy is represented exactly once.
    // Acceptance criteria: Three distinct identifiers exist and the rectangle area remains 5,000 mm².
    expect(new Set(instances.map((item) => item.instanceId)).size).toBe(3);
    expect(area(instances[0]!)).toBe(5000);
  });

  /**
   * Purpose: Protects grain and global rotation restrictions.
   * Description: Checks a free-grain part and a directional-grain part against global rotation.
   */
  it('applies rotation eligibility and grain restrictions', () => {
    const instance = expandParts([part()])[0]!;
    // Expected outcome: Rotation requires global, part-level, and grain permission.
    // Acceptance criteria: Free grain rotates, while global disablement and directional grain reject rotation.
    expect(canRotate(instance, true)).toBe(true);
    expect(canRotate(instance, false)).toBe(false);
    expect(canRotate({ ...instance, grainDirection: 'horizontal' }, true)).toBe(false);
  });

  /**
   * Purpose: Validates foundational rectangle operations used to prevent collisions.
   * Description: Checks intersection, splitting, and removal of contained free rectangles.
   */
  it('detects overlap and maintains free rectangles', () => {
    // Expected outcome: Overlap is detected and splitting preserves valid residual spaces.
    // Acceptance criteria: The overlapping pair intersects, splitting returns residuals, and a contained rectangle is pruned.
    expect(intersects({ x: 0, y: 0, width: 10, height: 10 }, { x: 5, y: 5, width: 10, height: 10 })).toBe(true);
    expect(splitFreeRectangle({ x: 0, y: 0, width: 100, height: 100 }, { x: 20, y: 20, width: 20, height: 20 })).toHaveLength(4);
    expect(
      pruneFreeRectangles([
        { x: 0, y: 0, width: 100, height: 100 },
        { x: 5, y: 5, width: 10, height: 10 }
      ])
    ).toHaveLength(1);
  });

  /**
   * Purpose: Protects deterministic placement and stock allocation behavior.
   * Description: Runs an identical multi-part, multi-sheet problem twice and compares its structural output.
   */
  it('places deterministically across multiple sheets and reports insufficient stock', () => {
    const input = [part({ quantity: 3, width: 180, height: 100 })];
    const stock = [sheet({ width: 200, height: 120, quantity: 2 })];
    const first = optimize(input, stock, settings);
    const second = optimize(input, stock, settings);
    // Expected outcome: Two sheets are opened, two pieces are placed, and one remains unplaced identically on every run.
    // Acceptance criteria: Counts match and the serialized geometry is stable after excluding elapsed duration.
    expect(first.layouts).toHaveLength(2);
    expect(first.unplacedParts).toHaveLength(1);
    expect(first.layouts).toEqual(second.layouts);
  });

  /**
   * Purpose: Ensures rotation can be the deciding factor in a valid placement.
   * Description: Fits a 120 by 180 part into a 200 by 140 sheet only after rotation and rejects it when rotation is disabled.
   */
  it('supports rotation-required placement and rotation-disabled rejection', () => {
    const stock = [sheet({ width: 220, height: 160 })];
    const rotatable = optimize([part({ width: 120, height: 180 })], stock, settings);
    const fixed = optimize([part({ width: 120, height: 180, canRotate: false })], stock, settings);
    // Expected outcome: The rotatable part fits while the fixed orientation does not.
    // Acceptance criteria: One rotated placement and one unplaced fixed part are returned.
    expect(rotatable.layouts[0]?.placedParts[0]?.rotated).toBe(true);
    expect(fixed.unplacedParts).toHaveLength(1);
  });

  /**
   * Purpose: Prevents incompatible material and thickness stock from being consumed.
   * Description: Attempts to place plywood against MDF and against a mismatched thickness.
   */
  it('groups material and thickness and ranks fewer unplaced parts first', () => {
    const materialMismatch = optimize([part()], [sheet({ material: 'MDF' })], settings);
    const thicknessMismatch = optimize([part()], [sheet({ thickness: 12 })], settings);
    const fitting = optimize([part()], [sheet()], settings);
    // Expected outcome: Only exactly compatible stock accepts the part, and that result ranks first.
    // Acceptance criteria: Mismatches remain unplaced and comparison prefers the fitting result.
    expect(materialMismatch.unplacedParts).toHaveLength(1);
    expect(thicknessMismatch.unplacedParts).toHaveLength(1);
    expect(compareResults(fitting, materialMismatch)).toBeLessThan(0);
  });
});
