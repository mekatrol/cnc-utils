import { computed, ref, watch } from 'vue';
import { defineStore } from 'pinia';
import { defaultSheets, type StockSheetDefinition } from '@/domain/models';
import { validateSheet } from '@/domain/validation';
import { makeId, readStored, writeStored } from '@/stores/storage';

const storageKey = 'cutlist-sheets-v1';
export const useSheetsStore = defineStore('sheets', () => {
  const sheets = ref<StockSheetDefinition[]>(readStored(storageKey, defaultSheets));
  const availableSheetCount = computed(() => sheets.value.reduce((total, sheet) => total + sheet.quantity, 0));
  const availableSheetArea = computed(() => sheets.value.reduce((total, sheet) => total + sheet.width * sheet.height * sheet.quantity, 0));
  const materialGroups = computed(() => new Set(sheets.value.map((sheet) => `${sheet.material}:${sheet.thickness}`)).size);
  const add = (sheet: StockSheetDefinition): boolean => {
    if (Object.keys(validateSheet(sheet)).length) return false;
    sheets.value.push({ ...sheet, id: sheet.id || makeId('sheet') });
    return true;
  };
  const edit = (sheet: StockSheetDefinition): boolean => {
    if (Object.keys(validateSheet(sheet)).length) return false;
    const index = sheets.value.findIndex((item) => item.id === sheet.id);
    if (index < 0) return false;
    sheets.value[index] = { ...sheet };
    return true;
  };
  const remove = (id: string): void => {
    sheets.value = sheets.value.filter((sheet) => sheet.id !== id);
  };
  const duplicate = (id: string): void => {
    const sheet = sheets.value.find((item) => item.id === id);
    if (sheet) sheets.value.push({ ...sheet, id: makeId('sheet'), name: `${sheet.name} copy` });
  };
  const clear = (): void => {
    sheets.value = [];
  };
  const reset = (): void => {
    sheets.value = structuredClone(defaultSheets);
  };
  const replace = (value: StockSheetDefinition[]): void => {
    sheets.value = structuredClone(value);
  };
  watch(sheets, (value) => writeStored(storageKey, value), { deep: true });
  return { sheets, availableSheetCount, availableSheetArea, materialGroups, add, edit, remove, duplicate, clear, reset, replace };
});
