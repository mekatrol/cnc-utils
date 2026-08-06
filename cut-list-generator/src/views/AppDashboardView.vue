<template>
  <section>
    <div class="eyebrow">Workshop planning</div>
    <h1>Make every sheet count.</h1>
    <p class="lede">Plan plywood cuts with deterministic, browser-only packing. Your project stays on this device.</p>
    <div class="metrics">
      <MetricCard label="Parts to cut" :value="parts.totalPartCount" /><MetricCard label="Stock sheets" :value="sheets.availableSheetCount" /><MetricCard
        label="Part area"
        :value="formatArea(parts.totalPartArea)"
      /><MetricCard label="Material groups" :value="parts.materialGroups" />
    </div>
    <div class="panel start">
      <div>
        <h2>Ready to find a layout?</h2>
        <p>Review stock and parts, then run the heuristic optimizer.</p>
      </div>
      <RouterLink class="button primary" to="/optimizer">Open optimizer</RouterLink>
    </div>
    <ProjectTransfer />
  </section>
</template>
<script setup lang="ts">
import { RouterLink } from 'vue-router';
import MetricCard from '@/components/shared/AppMetricCard.vue';
import ProjectTransfer from '@/components/shared/AppProjectTransfer.vue';
import { usePartsStore } from '@/stores/parts';
import { useSheetsStore } from '@/stores/sheets';
const parts = usePartsStore();
const sheets = useSheetsStore();
const formatArea = (value: number): string => `${(value / 1_000_000).toFixed(2)} m²`;
</script>
