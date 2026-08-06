<template>
  <form class="panel form-grid" data-testid="sheet-form" @submit.prevent="submit">
    <h2>{{ model.id ? 'Edit sheet' : 'Add a stock sheet' }}</h2>
    <label
      >Preset<select @change="applyPreset">
        <option value="">Custom</option>
        <option value="2400x1200">2400 × 1200 mm</option>
        <option value="2440x1220">2440 × 1220 mm</option>
      </select></label
    >
    <label
      >Name<input v-model.trim="model.name" data-testid="sheet-name" /><small v-if="errors.name">{{ errors.name }}</small></label
    >
    <label
      >Width (mm)<input v-model.number="model.width" type="number" min="1" data-testid="sheet-width" /><small v-if="errors.width">{{ errors.width }}</small></label
    >
    <label
      >Height (mm)<input v-model.number="model.height" type="number" min="1" data-testid="sheet-height" /><small v-if="errors.height">{{ errors.height }}</small></label
    >
    <label>Quantity<input v-model.number="model.quantity" type="number" min="1" data-testid="sheet-quantity" /></label>
    <label>Material<input v-model.trim="model.material" /></label><label>Thickness (mm)<input v-model.number="model.thickness" type="number" min="1" /></label
    ><label>Cost (optional)<input v-model.number="model.cost" type="number" min="0" /></label>
    <div class="actions">
      <button class="primary" data-testid="save-sheet">{{ model.id ? 'Save changes' : 'Add sheet' }}</button><button v-if="model.id" type="button" @click="$emit('cancel')">Cancel</button>
    </div>
  </form>
</template>
<script setup lang="ts">
import { reactive, watch } from 'vue';
import type { StockSheetDefinition, ValidationErrors } from '@/domain/models';
import { validateSheet } from '@/domain/validation';
const props = defineProps<{ sheet?: StockSheetDefinition }>();
const emit = defineEmits<{ save: [StockSheetDefinition]; cancel: [] }>();
const blank = (): StockSheetDefinition => ({ id: '', name: '', width: 2440, height: 1220, quantity: 1, material: 'Plywood', thickness: 18 });
const model = reactive<StockSheetDefinition>(blank());
const errors = reactive<ValidationErrors>({});
watch(
  () => props.sheet,
  (sheet) => Object.assign(model, sheet ? structuredClone(sheet) : blank()),
  { immediate: true }
);
const applyPreset = (event: Event): void => {
  const value = (event.target as HTMLSelectElement).value;
  if (value) {
    const [width, height] = value.split('x').map(Number);
    model.width = width!;
    model.height = height!;
    model.name = `${width} × ${height} plywood`;
  }
};
const submit = (): void => {
  Object.keys(errors).forEach((key) => delete errors[key]);
  Object.assign(errors, validateSheet(model));
  if (!Object.keys(errors).length) {
    emit('save', { ...model });
    if (!model.id) Object.assign(model, blank());
  }
};
</script>
