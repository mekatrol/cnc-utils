import { createRouter, createWebHistory } from 'vue-router';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', component: () => import('@/views/AppDashboardView.vue') },
    { path: '/parts', component: () => import('@/views/AppPartsView.vue') },
    { path: '/sheets', component: () => import('@/views/AppSheetsView.vue') },
    { path: '/optimizer', component: () => import('@/views/AppOptimizerView.vue') },
    { path: '/settings', component: () => import('@/views/AppSettingsView.vue') },
    { path: '/:pathMatch(.*)*', component: () => import('@/views/AppNotFoundView.vue') }
  ],
  scrollBehavior: () => ({ top: 0 })
});

export default router;
