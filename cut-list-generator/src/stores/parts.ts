import { computed, ref, watch } from 'vue';
import { defineStore } from 'pinia';
import { defaultParts, type PartDefinition } from '@/domain/models';
import { validatePart } from '@/domain/validation';
import { makeId, readStored, writeStored } from '@/stores/storage';

const storageKey = 'cutlist-parts-v1';
export const usePartsStore = defineStore('parts', () => {
  const parts = ref<PartDefinition[]>(readStored(storageKey, defaultParts));
  const totalPartCount = computed(() => parts.value.reduce((total, part) => total + part.quantity, 0));
  const totalPartArea = computed(() => parts.value.reduce((total, part) => total + part.width * part.height * part.quantity, 0));
  const materialGroups = computed(() => new Set(parts.value.map((part) => `${part.material}:${part.thickness}`)).size);
  const add = (part: PartDefinition): boolean => {
    if (Object.keys(validatePart(part)).length) return false;
    parts.value.push({ ...part, id: part.id || makeId('part') });
    return true;
  };
  const edit = (part: PartDefinition): boolean => {
    if (Object.keys(validatePart(part)).length) return false;
    const index = parts.value.findIndex((item) => item.id === part.id);
    if (index < 0) return false;
    parts.value[index] = { ...part };
    return true;
  };
  const remove = (id: string): void => {
    parts.value = parts.value.filter((part) => part.id !== id);
  };
  const duplicate = (id: string): void => {
    const part = parts.value.find((item) => item.id === id);
    if (part) parts.value.push({ ...part, id: makeId('part'), name: `${part.name} copy` });
  };
  const clear = (): void => {
    parts.value = [];
  };
  const reset = (): void => {
    parts.value = structuredClone(defaultParts);
  };
  const replace = (value: PartDefinition[]): void => {
    parts.value = structuredClone(value);
  };
  watch(parts, (value) => writeStored(storageKey, value), { deep: true });
  return { parts, totalPartCount, totalPartArea, materialGroups, add, edit, remove, duplicate, clear, reset, replace };
});
