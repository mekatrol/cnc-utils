import { mount } from '@vue/test-utils';
import { createPinia } from 'pinia';
import { describe, expect, it } from 'vitest';

import App from '@/App.vue';
import router from '@/router';

describe('App', () => {
  /**
   * Purpose: Protects the application's primary navigation shell.
   * Description: Mounts the root component with its production plugins and verifies the workshop brand is visible.
   */
  it('renders the application heading', () => {
    const wrapper = mount(App, { global: { plugins: [createPinia(), router] } });

    // Expected outcome: The mounted application identifies the cut-list product.
    // Acceptance criteria: The rendered text contains "PlyPlan" because that is the application shell's visible brand.
    expect(wrapper.text()).toContain('PlyPlan');
  });
});
