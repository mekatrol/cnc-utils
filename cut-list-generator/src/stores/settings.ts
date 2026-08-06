import { ref, watch } from 'vue';
import { defineStore } from 'pinia';
import { defaultSettings, type OptimizerSettings } from '@/domain/models';
import { readStored, writeStored } from '@/stores/storage';

const storageKey = 'cutlist-settings-v1';
export const useSettingsStore = defineStore('settings', () => {
  const settings = ref<OptimizerSettings>(readStored(storageKey, defaultSettings));
  const reset = (): void => {
    settings.value = structuredClone(defaultSettings);
  };
  const replace = (value: OptimizerSettings): void => {
    settings.value = structuredClone(value);
  };
  watch(settings, (value) => writeStored(storageKey, value), { deep: true });
  return { settings, reset, replace };
});
