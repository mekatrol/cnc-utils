<template>
  <section>
    <div class="page-title">
      <div>
        <div class="eyebrow">Heuristic nesting</div>
        <h1>Optimizer</h1>
        <p>Finds the best layout from deterministic strategies; it does not claim a mathematical optimum.</p>
      </div>
      <button class="primary run" data-testid="run-optimizer" :disabled="errors.length > 0 || results.running" @click="run">{{ results.running ? 'Working…' : 'Run optimizer' }}</button>
    </div>
    <div v-if="errors.length" class="notice error" role="alert">
      <strong>Check your inputs</strong>
      <ul>
        <li v-for="error in errors" :key="error">{{ error }}</li>
      </ul>
    </div>
    <template v-if="results.result">
      <div class="metrics">
        <MetricCard label="Sheets used" :value="results.result.layouts.length" /><MetricCard label="Parts placed" :value="placedCount" /><MetricCard
          label="Unplaced"
          :value="results.result.unplacedParts.length"
        /><MetricCard label="Utilization" :value="`${results.result.utilizationPercent.toFixed(1)}%`" />
      </div>
      <div class="panel summary">
        <div>
          <strong>Best layout found</strong>
          <p>{{ results.result.diagnostics.strategy }} · {{ results.result.diagnostics.ordering }} ordering · {{ results.result.durationMs.toFixed(1) }} ms</p>
          <p>{{ formatArea(results.result.totalPartArea) }} parts · {{ formatArea(results.result.totalWasteArea) }} waste</p>
        </div>
        <div class="actions">
          <button @click="showLabels = !showLabels">{{ showLabels ? 'Hide' : 'Show' }} labels</button><button @click="zoom = Math.min(1.5, zoom + 0.1)">Zoom in</button
          ><button @click="zoom = 1">Fit view</button><button @click="downloadJson('plyplan-result.json', results.result)">Export result</button><button @click="print">Print</button>
        </div>
      </div>
      <div v-if="results.result.unplacedParts.length" class="notice error" data-testid="unplaced-parts">
        <h2>Unplaced parts</h2>
        <ul>
          <li v-for="part in results.result.unplacedParts" :key="part.instanceId">{{ part.name }} ({{ part.width }} × {{ part.height }} mm) — {{ part.material }}, {{ part.thickness }} mm</li>
        </ul>
      </div>
      <SheetLayout
        v-for="(layout, index) in results.result.layouts"
        :key="layout.sheetInstanceId"
        :layout="layout"
        :index="index"
        :margin="settings.settings.edgeMargin"
        :show-labels="showLabels"
        :zoom="zoom"
      />
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import MetricCard from '@/components/shared/AppMetricCard.vue';
import SheetLayout from '@/components/results/AppSheetLayout.vue';
import { optimize } from '@/services/optimizer/optimizer';
import { usePartsStore } from '@/stores/parts';
import { useResultsStore } from '@/stores/results';
import { useSettingsStore } from '@/stores/settings';
import { useSheetsStore } from '@/stores/sheets';
import { downloadJson } from '@/utils/projectIo';
const parts = usePartsStore();
const sheets = useSheetsStore();
const settings = useSettingsStore();
const results = useResultsStore();
const showLabels = ref(true);
const zoom = ref(1);
const errors = computed(() => {
  const messages: string[] = [];
  if (!parts.parts.length) messages.push('Add at least one part.');
  if (!sheets.sheets.length) messages.push('Add at least one stock sheet.');
  if (sheets.sheets.some((sheet) => sheet.width <= settings.settings.edgeMargin * 2 || sheet.height <= settings.settings.edgeMargin * 2))
    messages.push('Edge margins leave no usable area on one or more sheets.');
  const groups = new Set(sheets.sheets.map((sheet) => `${sheet.material.toLowerCase()}:${sheet.thickness}`));
  parts.parts.forEach((part) => {
    if (!groups.has(`${part.material.toLowerCase()}:${part.thickness}`)) messages.push(`${part.name} has no compatible material and thickness stock.`);
  });
  return [...new Set(messages)];
});
const placedCount = computed(() => results.result?.layouts.reduce((total, layout) => total + layout.placedParts.length, 0) ?? 0);
const run = (): void => {
  results.running = true;
  results.setResult(optimize(parts.parts, sheets.sheets, settings.settings));
  results.running = false;
};
const formatArea = (value: number): string => `${(value / 1_000_000).toFixed(2)} m²`;
const print = (): void => window.print();
</script>
