import type { OptimizationResult, OptimizerSettings, PackingStrategy, PartDefinition, PartInstance, PlacedPart, Rect, SheetLayout, StockSheetDefinition } from '@/domain/models';
import { area, canRotate, pruneFreeRectangles, splitFreeRectangle } from '@/services/optimizer/geometry';

interface Candidate {
  free: Rect;
  height: number;
  rotated: boolean;
  score: number[];
  width: number;
}
interface SheetState {
  definition: StockSheetDefinition;
  free: Rect[];
  fromStock: boolean;
  id: string;
  placed: PlacedPart[];
}
const orderings = ['area', 'longest-side', 'perimeter', 'width', 'height'] as const;
type Ordering = (typeof orderings)[number];
const strategies: PackingStrategy[] = ['best-area-fit', 'best-short-side-fit', 'best-long-side-fit', 'bottom-left'];
const key = (material: string, thickness: number): string => `${material.trim().toLocaleLowerCase()}::${thickness}`;

export const expandParts = (parts: PartDefinition[]): PartInstance[] =>
  parts.flatMap((part) => Array.from({ length: part.quantity }, (_, index) => ({ ...part, instanceId: `${part.id}-${index + 1}` })));
const sortParts = (parts: PartInstance[], ordering: Ordering): PartInstance[] =>
  [...parts].sort((left, right) => {
    const metric = (part: PartInstance): number =>
      ordering === 'area'
        ? part.width * part.height
        : ordering === 'longest-side'
          ? Math.max(part.width, part.height)
          : ordering === 'perimeter'
            ? part.width + part.height
            : ordering === 'width'
              ? part.width
              : part.height;
    return metric(right) - metric(left) || left.instanceId.localeCompare(right.instanceId);
  });
const candidateScore = (free: Rect, width: number, height: number, strategy: PackingStrategy): number[] => {
  const horizontal = free.width - width;
  const vertical = free.height - height;
  if (strategy === 'bottom-left') return [free.y, free.x];
  if (strategy === 'best-short-side-fit') return [Math.min(horizontal, vertical), Math.max(horizontal, vertical), free.y, free.x];
  if (strategy === 'best-long-side-fit') return [Math.max(horizontal, vertical), Math.min(horizontal, vertical), free.y, free.x];
  return [free.width * free.height - width * height, Math.min(horizontal, vertical), free.y, free.x];
};
const compareScore = (left: number[], right: number[]): number =>
  left.find((value, index) => value !== right[index]) === undefined ? 0 : left.find((value, index) => value !== right[index])! - right[left.findIndex((value, index) => value !== right[index])]!;

const findCandidate = (sheet: SheetState, part: PartInstance, settings: OptimizerSettings, strategy: PackingStrategy): Candidate | undefined => {
  const orientations = [
    { width: part.width, height: part.height, rotated: false },
    ...(canRotate(part, settings.allowRotation) && part.width !== part.height ? [{ width: part.height, height: part.width, rotated: true }] : [])
  ];
  const candidates = sheet.free.flatMap((free) =>
    orientations
      .filter((orientation) => orientation.width <= free.width && orientation.height <= free.height)
      .map((orientation) => ({ ...orientation, free, score: candidateScore(free, orientation.width, orientation.height, strategy) }))
  );
  return candidates.sort((left, right) => compareScore(left.score, right.score) || Number(left.rotated) - Number(right.rotated))[0];
};

/* Kerf convention: free space reserves kerf on a part's right/top, except when the physical part touches the usable outer boundary. SVG output always uses physical dimensions. */
const place = (sheet: SheetState, part: PartInstance, candidate: Candidate, settings: OptimizerSettings): void => {
  const boundaryX = settings.edgeMargin + sheet.definition.width - settings.edgeMargin * 2;
  const boundaryY = settings.edgeMargin + sheet.definition.height - settings.edgeMargin * 2;
  const reserved: Rect = {
    x: candidate.free.x,
    y: candidate.free.y,
    width: candidate.width + (candidate.free.x + candidate.width < boundaryX ? settings.kerf : 0),
    height: candidate.height + (candidate.free.y + candidate.height < boundaryY ? settings.kerf : 0)
  };
  sheet.placed.push({ instanceId: part.instanceId, partId: part.id, partName: part.name, x: reserved.x, y: reserved.y, width: candidate.width, height: candidate.height, rotated: candidate.rotated });
  sheet.free = pruneFreeRectangles(sheet.free.flatMap((free) => splitFreeRectangle(free, reserved)));
};
const createSheet = (definition: StockSheetDefinition, index: number, settings: OptimizerSettings, fromStock = true): SheetState => ({
  definition,
  id: `${definition.id}-${index + 1}`,
  fromStock,
  placed: [],
  free: [{ x: settings.edgeMargin, y: settings.edgeMargin, width: definition.width - settings.edgeMargin * 2, height: definition.height - settings.edgeMargin * 2 }]
});
const compareResults = (left: OptimizationResult, right: OptimizationResult): number =>
  left.unplacedParts.length - right.unplacedParts.length ||
  left.layouts.length - right.layouts.length ||
  left.totalWasteArea - right.totalWasteArea ||
  right.utilizationPercent - left.utilizationPercent ||
  `${left.diagnostics.strategy}:${left.diagnostics.ordering}`.localeCompare(`${right.diagnostics.strategy}:${right.diagnostics.ordering}`);
export { compareResults };

const runAttempt = (parts: PartInstance[], sheets: StockSheetDefinition[], settings: OptimizerSettings, strategy: PackingStrategy, ordering: Ordering): OptimizationResult => {
  const started = performance.now();
  const available = sheets
    .flatMap((sheet) => Array.from({ length: sheet.quantity }, (_, index) => createSheet(sheet, index, settings)))
    .filter((sheet) => sheet.free[0]!.width > 0 && sheet.free[0]!.height > 0);
  const opened: SheetState[] = [];
  const orderedCounts = new Map<string, number>();
  const unplaced: PartInstance[] = [];
  sortParts(parts, ordering).forEach((part) => {
    const compatible = (sheet: SheetState): boolean => key(sheet.definition.material, sheet.definition.thickness) === key(part.material, part.thickness);
    let target = opened.map((sheet) => ({ sheet, candidate: compatible(sheet) ? findCandidate(sheet, part, settings, strategy) : undefined })).find((entry) => entry.candidate);
    if (!target) {
      const nextIndex = available.findIndex((sheet) => compatible(sheet) && findCandidate(sheet, part, settings, strategy));
      if (nextIndex >= 0) {
        const sheet = available.splice(nextIndex, 1)[0]!;
        opened.push(sheet);
        target = { sheet, candidate: findCandidate(sheet, part, settings, strategy) };
      }
    }
    if (!target && settings.allowAdditionalSheets) {
      const definition = sheets.find((sheet) => {
        if (key(sheet.material, sheet.thickness) !== key(part.material, part.thickness)) return false;
        const trial = createSheet(sheet, sheet.quantity, settings, false);
        return trial.free[0]!.width > 0 && trial.free[0]!.height > 0 && findCandidate(trial, part, settings, strategy) !== undefined;
      });
      if (definition) {
        const orderedCount = orderedCounts.get(definition.id) ?? 0;
        orderedCounts.set(definition.id, orderedCount + 1);
        const sheet = createSheet(definition, definition.quantity + orderedCount, settings, false);
        opened.push(sheet);
        target = { sheet, candidate: findCandidate(sheet, part, settings, strategy) };
      }
    }
    if (target?.candidate) place(target.sheet, part, target.candidate, settings);
    else unplaced.push(part);
  });
  const layouts: SheetLayout[] = opened.map((sheet) => {
    const usedArea = sheet.placed.reduce((total, part) => total + part.width * part.height, 0);
    const sheetArea = area(sheet.definition);
    return {
      sheetInstanceId: sheet.id,
      stockSheetId: sheet.definition.id,
      width: sheet.definition.width,
      height: sheet.definition.height,
      placedParts: sheet.placed,
      usedArea,
      wasteArea: sheetArea - usedArea,
      utilizationPercent: sheetArea ? (usedArea / sheetArea) * 100 : 0
    };
  });
  const totalSheetArea = layouts.reduce((total, layout) => total + layout.width * layout.height, 0);
  const totalPartArea = layouts.reduce((total, layout) => total + layout.usedArea, 0);
  const sheetsUsedFromStock = opened.filter((sheet) => sheet.fromStock).length;
  const sheetsToOrder = opened.length - sheetsUsedFromStock;
  return {
    layouts,
    unplacedParts: unplaced,
    sheetsUsedFromStock,
    sheetsToOrder,
    totalSheetArea,
    totalPartArea,
    totalWasteArea: totalSheetArea - totalPartArea,
    utilizationPercent: totalSheetArea ? (totalPartArea / totalSheetArea) * 100 : 0,
    durationMs: performance.now() - started,
    diagnostics: { strategy, ordering, warnings: unplaced.length ? [`${unplaced.length} part(s) could not be placed.`] : [] }
  };
};

export const optimize = (parts: PartDefinition[], sheets: StockSheetDefinition[], settings: OptimizerSettings): OptimizationResult => {
  const instances = expandParts(parts);
  const attempts: OptimizationResult[] = [];
  const selectedStrategies = [settings.strategy, ...strategies.filter((strategy) => strategy !== settings.strategy)];
  for (const strategy of selectedStrategies)
    for (const ordering of orderings) {
      if (attempts.length >= settings.maxIterations) break;
      attempts.push(runAttempt(instances, sheets, settings, strategy, ordering));
    }
  return attempts.sort(compareResults)[0] ?? runAttempt(instances, sheets, settings, settings.strategy, 'area');
};
