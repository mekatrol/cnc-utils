<template>
  <article class="metric">
    <span>{{ label }}</span
    ><strong>{{ value }}</strong>
  </article>
</template>
<script setup lang="ts">
defineProps<{ label: string; value: string | number }>();
</script>
