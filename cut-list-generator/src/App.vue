<template><AppShell automation="app-shell" /></template>
<script setup lang="ts">
import AppShell from '@/components/layout/AppShell.vue';
</script>
<style>
:root {
  --ink: #20231f;
  --muted: #697067;
  --paper: #f3f0e8;
  --panel: #fffcf5;
  --line: #d8d2c4;
  --green: #315b45;
  --lime: #d5e85a;
  --danger: #9a3f32;
  font-family: Manrope, sans-serif;
  color: var(--ink);
  background: var(--paper);
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  min-width: 320px;
}
button,
input,
select {
  font: inherit;
}
button,
.button {
  border: 1px solid var(--line);
  background: white;
  padding: 0.65rem 0.9rem;
  border-radius: 6px;
  cursor: pointer;
  color: inherit;
  text-decoration: none;
}
button:hover,
.button:hover {
  border-color: var(--green);
}
button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.primary {
  background: var(--green);
  color: white;
  border-color: var(--green);
}
.danger {
  color: var(--danger);
}
.app-header {
  min-height: 76px;
  padding: 1rem clamp(1rem, 4vw, 4rem);
  background: #1f2923;
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 2rem;
}
.brand {
  color: white;
  text-decoration: none;
  font-size: 1.35rem;
  font-weight: 800;
}
.brand span {
  color: #aab7ad;
  font-size: 0.65rem;
  display: block;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}
.app-header nav {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}
.app-header nav a {
  color: #dce4dd;
  text-decoration: none;
  padding: 0.55rem 0.7rem;
  border-radius: 5px;
  font-size: 0.9rem;
}
.app-header nav a.router-link-active {
  color: #1f2923;
  background: var(--lime);
}
main {
  max-width: 1320px;
  margin: 0 auto;
  padding: clamp(1.5rem, 4vw, 4rem);
}
h1 {
  font-size: clamp(2.2rem, 5vw, 4.8rem);
  line-height: 1;
  margin: 0.35rem 0 1rem;
  letter-spacing: -0.05em;
}
h2,
h3 {
  margin-top: 0;
}
.eyebrow {
  font:
    500 0.75rem DM Mono,
    monospace;
  color: var(--green);
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.lede {
  max-width: 700px;
  font-size: 1.2rem;
  color: var(--muted);
}
.panel {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 1.3rem;
  box-shadow: 0 3px 0 rgba(38, 45, 39, 0.04);
}
.metrics {
  margin: 2rem 0;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem;
}
.metric span {
  color: var(--muted);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}
.metric strong {
  display: block;
  margin-top: 0.4rem;
  font:
    500 1.65rem DM Mono,
    monospace;
}
.start,
.transfer,
.summary,
.page-title,
.layout-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}
.transfer {
  margin-top: 1rem;
  flex-wrap: wrap;
}
.page-title {
  margin-bottom: 1.5rem;
}
.page-title h1 {
  margin-bottom: 0.3rem;
}
.page-title p {
  color: var(--muted);
}
.form-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.form-grid h2,
.form-grid .actions {
  grid-column: 1 / -1;
}
.form-grid label {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-weight: 600;
  font-size: 0.85rem;
}
.form-grid input,
.form-grid select {
  background: white;
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 0.65rem;
  min-width: 0;
}
.form-grid .check {
  flex-direction: row;
  align-items: center;
}
.form-grid small {
  color: var(--danger);
}
.actions {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.table-wrap {
  overflow-x: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th {
  text-align: left;
  color: var(--muted);
  font-size: 0.72rem;
  text-transform: uppercase;
}
th,
td {
  border-bottom: 1px solid var(--line);
  padding: 0.9rem 0.5rem;
}
.card-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
}
.sheet-chip {
  font:
    500 1.4rem DM Mono,
    monospace;
  color: var(--green);
}
.notice {
  padding: 1rem 1.2rem;
  border-left: 4px solid var(--green);
  background: #edf2e9;
  margin: 1rem 0;
}
.notice.error {
  border-color: var(--danger);
  background: #f5e9e5;
}
.layout-card {
  margin-top: 1.5rem;
  overflow: hidden;
}
.layout-card svg {
  width: 100%;
  max-height: 620px;
  margin-top: 1rem;
  transform-origin: top left;
}
.sheet {
  fill: #e6dfd0;
  stroke: #20231f;
  stroke-width: 5;
}
.usable {
  fill: none;
  stroke: #777;
  stroke-width: 2;
  stroke-dasharray: 12;
}
.part {
  fill: #b9cf68;
  stroke: #315b45;
  stroke-width: 3;
}
.layout-card text {
  font:
    500 22px DM Mono,
    monospace;
  fill: #17231a;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
}
.file {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
}
.empty {
  text-align: center;
  padding: 10vh 0;
}
@media (max-width: 800px) {
  .app-header {
    align-items: flex-start;
    flex-direction: column;
  }
  .metrics,
  .form-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  .card-grid {
    grid-template-columns: 1fr;
  }
  .page-title {
    align-items: flex-start;
  }
}
@media (max-width: 500px) {
  .metrics,
  .form-grid {
    grid-template-columns: 1fr;
  }
  .app-header nav {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
  }
  .start,
  .summary,
  .page-title {
    flex-direction: column;
    align-items: stretch;
  }
}
@media print {
  .app-header,
  button,
  .actions,
  .notice.error {
    display: none !important;
  }
  main {
    padding: 0;
  }
  .panel {
    box-shadow: none;
    break-inside: avoid;
  }
}
</style>
