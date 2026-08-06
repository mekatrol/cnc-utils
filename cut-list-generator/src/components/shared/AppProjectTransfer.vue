<template>
  <div class="panel transfer">
    <div>
      <h2>Project data</h2>
      <p>Export a portable JSON file or restore a validated project.</p>
    </div>
    <div class="actions">
      <button data-testid="export-project" @click="exportProject">Export project</button
      ><label class="button">Import project<input class="file" type="file" accept="application/json" data-testid="import-project" @change="importProject" /></label>
    </div>
    <p v-if="message" role="status">{{ message }}</p>
  </div>
</template>
<script setup lang="ts">
import { ref } from 'vue';
import { usePartsStore } from '@/stores/parts';
import { useSettingsStore } from '@/stores/settings';
import { useSheetsStore } from '@/stores/sheets';
import { downloadJson, parseProject } from '@/utils/projectIo';
const parts = usePartsStore();
const sheets = useSheetsStore();
const settings = useSettingsStore();
const message = ref('');
const exportProject = (): void => downloadJson('plyplan-project.json', { version: 1, setCount: parts.setCount, parts: parts.parts, sheets: sheets.sheets, settings: settings.settings });
const importProject = async (event: Event): Promise<void> => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    const project = parseProject(await file.text());
    parts.replace(project.parts);
    parts.setCount = project.setCount;
    sheets.replace(project.sheets);
    settings.replace(project.settings);
    message.value = 'Project imported successfully.';
  } catch (error) {
    message.value = error instanceof Error ? `Import rejected: ${error.message}` : 'Import rejected.';
  }
};
</script>
