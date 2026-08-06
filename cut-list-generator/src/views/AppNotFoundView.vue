<template>
  <section class="empty">
    <div class="eyebrow">404</div>
    <h1>That board isn't in the rack.</h1>
    <p>The page you requested could not be found.</p>
    <RouterLink class="button primary" to="/">Back to dashboard</RouterLink>
  </section>
</template>
<script setup lang="ts">
import { RouterLink } from 'vue-router';
</script>
