import { ref } from 'vue';
import { defineStore } from 'pinia';
import type { OptimizationResult } from '@/domain/models';
export const useResultsStore = defineStore('results', () => {
  const result = ref<OptimizationResult>();
  const running = ref(false);
  const setResult = (value: OptimizationResult): void => {
    result.value = value;
  };
  const clear = (): void => {
    result.value = undefined;
  };
  return { result, running, setResult, clear };
});
