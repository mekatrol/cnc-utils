<template>
  <header class="app-header">
    <RouterLink class="brand" to="/">PlyPlan <span>Cut-list optimizer</span></RouterLink>
    <nav aria-label="Primary navigation">
      <RouterLink to="/">Dashboard</RouterLink><RouterLink to="/parts">Parts</RouterLink><RouterLink to="/sheets">Sheets</RouterLink><RouterLink to="/optimizer">Optimize</RouterLink
      ><RouterLink to="/settings">Settings</RouterLink>
    </nav>
  </header>
  <main><RouterView /></main>
</template>

<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router';
defineProps<{ automation: string }>();
</script>
