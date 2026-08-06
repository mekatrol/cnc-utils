<template>
  <article class="panel layout-card" :data-testid="`sheet-layout-${index}`">
    <div class="layout-title">
      <h3>Sheet {{ index + 1 }}</h3>
      <span>{{ layout.utilizationPercent.toFixed(1) }}% used · {{ formatArea(layout.wasteArea) }} waste</span>
    </div>
    <svg :viewBox="`0 0 ${layout.width} ${layout.height}`" role="img" :aria-labelledby="titleId" :style="{ transform: `scale(${zoom})` }">
      <title :id="titleId">Sheet {{ index + 1 }} with {{ layout.placedParts.length }} placed parts</title>
      <rect class="sheet" x="0" y="0" :width="layout.width" :height="layout.height" />
      <rect class="usable" :x="margin" :y="margin" :width="layout.width - margin * 2" :height="layout.height - margin * 2" />
      <g v-for="part in layout.placedParts" :key="part.instanceId" :data-testid="`placed-part-${part.instanceId}`">
        <rect class="part" :x="part.x" :y="part.y" :width="part.width" :height="part.height" />
        <text v-if="showLabels" :x="part.x + 12" :y="part.y + 25">{{ part.partName }} · {{ part.width }} × {{ part.height }}{{ part.rotated ? ' ↻' : '' }}</text>
      </g>
    </svg>
    <p class="sr-only">Placed parts: {{ layout.placedParts.map((part) => `${part.partName}, ${part.width} by ${part.height} millimetres`).join('; ') }}</p>
  </article>
</template>
<script setup lang="ts">
import { computed } from 'vue';
import type { SheetLayout } from '@/domain/models';
const props = defineProps<{ index: number; layout: SheetLayout; margin: number; showLabels: boolean; zoom: number }>();
const titleId = computed(() => `sheet-title-${props.index}`);
const formatArea = (value: number): string => `${(value / 1_000_000).toFixed(2)} m²`;
</script>
