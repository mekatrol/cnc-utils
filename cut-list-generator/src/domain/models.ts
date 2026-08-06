export type GrainDirection = 'none' | 'horizontal' | 'vertical';
export type PackingStrategy = 'best-area-fit' | 'best-short-side-fit' | 'best-long-side-fit' | 'bottom-left';

export interface PartDefinition {
  id: string;
  name: string;
  width: number;
  height: number;
  quantity: number;
  canRotate: boolean;
  grainDirection: GrainDirection;
  material: string;
  thickness: number;
}
export interface StockSheetDefinition {
  id: string;
  name: string;
  width: number;
  height: number;
  quantity: number;
  material: string;
  thickness: number;
  cost?: number;
}
export interface OptimizerSettings {
  kerf: number;
  edgeMargin: number;
  allowRotation: boolean;
  strategy: PackingStrategy;
  maxIterations: number;
}
export interface PartInstance extends Omit<PartDefinition, 'quantity'> {
  instanceId: string;
}
export interface PlacedPart {
  instanceId: string;
  partId: string;
  partName: string;
  x: number;
  y: number;
  width: number;
  height: number;
  rotated: boolean;
}
export interface SheetLayout {
  sheetInstanceId: string;
  stockSheetId: string;
  width: number;
  height: number;
  placedParts: PlacedPart[];
  usedArea: number;
  wasteArea: number;
  utilizationPercent: number;
}
export interface OptimizationDiagnostics {
  strategy: PackingStrategy;
  ordering: string;
  warnings: string[];
}
export interface OptimizationResult {
  layouts: SheetLayout[];
  unplacedParts: PartInstance[];
  totalSheetArea: number;
  totalPartArea: number;
  totalWasteArea: number;
  utilizationPercent: number;
  durationMs: number;
  diagnostics: OptimizationDiagnostics;
}
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}
export interface ProjectData {
  version: number;
  setCount: number;
  parts: PartDefinition[];
  sheets: StockSheetDefinition[];
  settings: OptimizerSettings;
}
export interface ValidationErrors {
  [field: string]: string;
}

export const defaultParts: PartDefinition[] = [
  { id: 'side-panel', name: 'Side Panel', width: 720, height: 500, quantity: 2, canRotate: true, grainDirection: 'none', material: 'Plywood', thickness: 18 },
  { id: 'shelf', name: 'Shelf', width: 800, height: 300, quantity: 4, canRotate: true, grainDirection: 'none', material: 'Plywood', thickness: 18 },
  { id: 'base', name: 'Base', width: 800, height: 500, quantity: 1, canRotate: true, grainDirection: 'none', material: 'Plywood', thickness: 18 },
  { id: 'back-rail', name: 'Back Rail', width: 800, height: 120, quantity: 2, canRotate: true, grainDirection: 'none', material: 'Plywood', thickness: 18 }
];
export const defaultSheets: StockSheetDefinition[] = [{ id: 'standard-sheet', name: 'Standard 2440 × 1220', width: 2440, height: 1220, quantity: 2, material: 'Plywood', thickness: 18, cost: 90 }];
export const defaultSettings: OptimizerSettings = { kerf: 3, edgeMargin: 10, allowRotation: true, strategy: 'best-area-fit', maxIterations: 20 };
