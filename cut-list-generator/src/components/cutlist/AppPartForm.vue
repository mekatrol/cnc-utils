<template>
  <form class="panel form-grid" data-testid="part-form" @submit.prevent="submit">
    <h2>{{ model.id ? 'Edit part' : 'Add a part' }}</h2>
    <label
      >Name<input v-model.trim="model.name" data-testid="part-name" /><small v-if="errors.name">{{ errors.name }}</small></label
    >
    <label
      >Width (mm)<input v-model.number="model.width" type="number" min="1" data-testid="part-width" /><small v-if="errors.width">{{ errors.width }}</small></label
    >
    <label
      >Height (mm)<input v-model.number="model.height" type="number" min="1" data-testid="part-height" /><small v-if="errors.height">{{ errors.height }}</small></label
    >
    <label
      >Quantity<input v-model.number="model.quantity" type="number" min="1" data-testid="part-quantity" /><small v-if="errors.quantity">{{ errors.quantity }}</small></label
    >
    <label>Material<input v-model.trim="model.material" /></label>
    <label
      >Thickness (mm)<input v-model.number="model.thickness" type="number" min="1" /><small v-if="errors.thickness">{{ errors.thickness }}</small></label
    >
    <label
      >Grain direction<select v-model="model.grainDirection">
        <option value="none">None</option>
        <option value="horizontal">Horizontal</option>
        <option value="vertical">Vertical</option>
      </select></label
    >
    <label class="check"><input v-model="model.canRotate" type="checkbox" /> Allow rotation</label>
    <div class="actions">
      <button class="primary" data-testid="save-part">{{ model.id ? 'Save changes' : 'Add part' }}</button><button v-if="model.id" type="button" @click="$emit('cancel')">Cancel</button>
    </div>
  </form>
</template>
<script setup lang="ts">
import { reactive, watch } from 'vue';
import type { PartDefinition, ValidationErrors } from '@/domain/models';
import { validatePart } from '@/domain/validation';
const props = defineProps<{ part?: PartDefinition }>();
const emit = defineEmits<{ save: [PartDefinition]; cancel: [] }>();
const blank = (): PartDefinition => ({ id: '', name: '', width: 600, height: 300, quantity: 1, canRotate: true, grainDirection: 'none', material: 'Plywood', thickness: 18 });
const model = reactive<PartDefinition>(blank());
const errors = reactive<ValidationErrors>({});
watch(
  () => props.part,
  (part) => Object.assign(model, part ? structuredClone(part) : blank()),
  { immediate: true }
);
const submit = (): void => {
  Object.keys(errors).forEach((key) => delete errors[key]);
  Object.assign(errors, validatePart(model));
  if (!Object.keys(errors).length) {
    emit('save', { ...model });
    if (!model.id) Object.assign(model, blank());
  }
};
</script>
