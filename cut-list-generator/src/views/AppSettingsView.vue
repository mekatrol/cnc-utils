<template>
  <section>
    <div class="eyebrow">Machine setup</div>
    <h1>Cutting settings</h1>
    <div class="panel form-grid settings">
      <label
        >Blade kerf (mm)<input v-model.number="store.settings.kerf" type="number" min="0" data-testid="kerf-input" /><small
          >Reserved on the right and top of each part, except at the usable outer boundary.</small
        ></label
      ><label>Edge margin (mm)<input v-model.number="store.settings.edgeMargin" type="number" min="0" data-testid="margin-input" /></label
      ><label
        >Preferred heuristic<select v-model="store.settings.strategy">
          <option value="best-area-fit">Best area fit</option>
          <option value="best-short-side-fit">Best short-side fit</option>
          <option value="best-long-side-fit">Best long-side fit</option>
          <option value="bottom-left">Bottom left</option>
        </select></label
      ><label>Maximum attempts<input v-model.number="store.settings.maxIterations" type="number" min="1" max="20" /></label
      ><label class="check"><input v-model="store.settings.allowRotation" type="checkbox" /> Allow rotation globally</label
      ><label class="check"><input v-model="store.settings.allowAdditionalSheets" type="checkbox" /> Use as many additional sheets as needed</label>
      <div class="actions"><button @click="store.reset">Reset settings</button></div>
    </div>
  </section>
</template>
<script setup lang="ts">
import { useSettingsStore } from '@/stores/settings';
const store = useSettingsStore();
</script>
