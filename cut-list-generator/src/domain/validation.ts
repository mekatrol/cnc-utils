import type { PartDefinition, StockSheetDefinition, ValidationErrors } from '@/domain/models';

export const validatePart = (part: PartDefinition): ValidationErrors => {
  const errors: ValidationErrors = {};
  if (!part.name.trim()) errors.name = 'Name is required.';
  if (part.width <= 0) errors.width = 'Width must be greater than zero.';
  if (part.height <= 0) errors.height = 'Height must be greater than zero.';
  if (!Number.isInteger(part.quantity) || part.quantity <= 0) errors.quantity = 'Quantity must be a positive integer.';
  if (part.thickness <= 0) errors.thickness = 'Thickness must be greater than zero.';
  return errors;
};

export const validateSheet = (sheet: StockSheetDefinition): ValidationErrors => {
  const errors: ValidationErrors = {};
  if (!sheet.name.trim()) errors.name = 'Name is required.';
  if (sheet.width <= 0) errors.width = 'Width must be greater than zero.';
  if (sheet.height <= 0) errors.height = 'Height must be greater than zero.';
  if (!Number.isInteger(sheet.quantity) || sheet.quantity <= 0) errors.quantity = 'Quantity must be a positive integer.';
  if (sheet.thickness <= 0) errors.thickness = 'Thickness must be greater than zero.';
  if (sheet.cost !== undefined && sheet.cost < 0) errors.cost = 'Cost cannot be negative.';
  return errors;
};
