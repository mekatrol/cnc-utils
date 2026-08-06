<template>
  <section>
    <div class="page-title">
      <div>
        <div class="eyebrow">Cut list</div>
        <h1>Parts</h1>
        <p>{{ store.totalPartCount }} pieces · {{ (store.totalPartArea / 1_000_000).toFixed(2) }} m²</p>
      </div>
      <button class="danger" @click="clear">Clear all</button>
    </div>
    <div class="panel set-control">
      <div>
        <h2>Production sets</h2>
        <p>Each part quantity is per set. Increase this to make multiple identical cupboards or assemblies.</p>
      </div>
      <label for="set-count">Number of sets<input id="set-count" v-model.number="store.setCount" type="number" min="1" step="1" data-testid="set-count" @change="validateSetCount" /></label>
    </div>
    <PartForm :part="editing" @save="save" @cancel="editing = undefined" />
    <div class="panel table-wrap">
      <table>
        <thead>
          <tr>
            <th>Part</th>
            <th>Dimensions</th>
            <th>Qty per set</th>
            <th>Total</th>
            <th>Material</th>
            <th>Grain</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="part in store.parts" :key="part.id" :data-testid="`part-row-${part.id}`">
            <td>
              <strong>{{ part.name }}</strong>
            </td>
            <td>{{ part.width }} × {{ part.height }} × {{ part.thickness }} mm</td>
            <td>{{ part.quantity }}</td>
            <td>{{ part.quantity * store.setCount }}</td>
            <td>{{ part.material }}</td>
            <td>{{ part.grainDirection }}</td>
            <td class="actions">
              <button @click="editing = part">Edit</button><button @click="store.duplicate(part.id)">Duplicate</button><button class="danger" @click="store.remove(part.id)">Delete</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
<script setup lang="ts">
import { ref } from 'vue';
import PartForm from '@/components/cutlist/AppPartForm.vue';
import type { PartDefinition } from '@/domain/models';
import { usePartsStore } from '@/stores/parts';
const store = usePartsStore();
const editing = ref<PartDefinition>();
const save = (part: PartDefinition): void => {
  if (part.id) store.edit(part);
  else store.add(part);
  editing.value = undefined;
};
const clear = (): void => {
  if (confirm('Remove every part?')) store.clear();
};
const validateSetCount = (): void => {
  if (!Number.isInteger(store.setCount) || store.setCount < 1) store.setCount = 1;
};
</script>

<style scoped>
.set-control {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.set-control h2,
.set-control p {
  margin-bottom: 0.25rem;
}
.set-control label {
  display: grid;
  gap: 0.35rem;
  font-weight: 700;
}
.set-control input {
  width: 9rem;
  padding: 0.65rem;
  border: 1px solid var(--line);
  border-radius: 5px;
}
@media (max-width: 600px) {
  .set-control {
    align-items: stretch;
    flex-direction: column;
  }
}
</style>
