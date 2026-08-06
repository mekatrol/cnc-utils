import type { GrainDirection, PartInstance, Rect } from '@/domain/models';

export const area = (rectangle: Pick<Rect, 'width' | 'height'>): number => rectangle.width * rectangle.height;
export const intersects = (left: Rect, right: Rect): boolean => left.x < right.x + right.width && left.x + left.width > right.x && left.y < right.y + right.height && left.y + left.height > right.y;
export const contains = (outer: Rect, inner: Rect): boolean =>
  inner.x >= outer.x && inner.y >= outer.y && inner.x + inner.width <= outer.x + outer.width && inner.y + inner.height <= outer.y + outer.height;
export const grainAllowsRotation = (grain: GrainDirection): boolean => grain === 'none';
export const canRotate = (part: PartInstance, globalRotation: boolean): boolean => globalRotation && part.canRotate && grainAllowsRotation(part.grainDirection);

export const splitFreeRectangle = (free: Rect, used: Rect): Rect[] => {
  if (!intersects(free, used)) return [free];
  const rectangles: Rect[] = [];
  if (used.x > free.x) rectangles.push({ x: free.x, y: free.y, width: used.x - free.x, height: free.height });
  if (used.x + used.width < free.x + free.width) rectangles.push({ x: used.x + used.width, y: free.y, width: free.x + free.width - used.x - used.width, height: free.height });
  if (used.y > free.y) rectangles.push({ x: free.x, y: free.y, width: free.width, height: used.y - free.y });
  if (used.y + used.height < free.y + free.height) rectangles.push({ x: free.x, y: used.y + used.height, width: free.width, height: free.y + free.height - used.y - used.height });
  return rectangles.filter((rectangle) => rectangle.width > 0 && rectangle.height > 0);
};

export const pruneFreeRectangles = (rectangles: Rect[]): Rect[] => rectangles.filter((rectangle, index) => !rectangles.some((other, otherIndex) => index !== otherIndex && contains(other, rectangle)));
