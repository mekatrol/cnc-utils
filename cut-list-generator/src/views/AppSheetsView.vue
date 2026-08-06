<template>
  <section>
    <div class="page-title">
      <div>
        <div class="eyebrow">Stock inventory</div>
        <h1>Plywood sheets</h1>
        <p>{{ store.availableSheetCount }} sheets available</p>
      </div>
      <button class="danger" @click="clear">Clear all</button>
    </div>
    <SheetForm :sheet="editing" @save="save" @cancel="editing = undefined" />
    <div class="card-grid">
      <article v-for="sheet in store.sheets" :key="sheet.id" class="panel" :data-testid="`sheet-row-${sheet.id}`">
        <h2>{{ sheet.name }}</h2>
        <div class="sheet-chip">{{ sheet.width }} × {{ sheet.height }} mm</div>
        <p>{{ sheet.quantity }} in stock · {{ sheet.material }} · {{ sheet.thickness }} mm</p>
        <p v-if="sheet.cost !== undefined">{{ new Intl.NumberFormat(undefined, { style: 'currency', currency: 'AUD' }).format(sheet.cost) }} each</p>
        <div class="actions">
          <button @click="editing = sheet">Edit</button><button @click="store.duplicate(sheet.id)">Duplicate</button><button class="danger" @click="store.remove(sheet.id)">Delete</button>
        </div>
      </article>
    </div>
  </section>
</template>
<script setup lang="ts">
import { ref } from 'vue';
import SheetForm from '@/components/sheets/AppSheetForm.vue';
import type { StockSheetDefinition } from '@/domain/models';
import { useSheetsStore } from '@/stores/sheets';
const store = useSheetsStore();
const editing = ref<StockSheetDefinition>();
const save = (sheet: StockSheetDefinition): void => {
  if (sheet.id) store.edit(sheet);
  else store.add(sheet);
  editing.value = undefined;
};
const clear = (): void => {
  if (confirm('Remove every stock sheet?')) store.clear();
};
</script>
