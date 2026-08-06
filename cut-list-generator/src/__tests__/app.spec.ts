import { mount } from '@vue/test-utils';
import { describe, expect, it } from 'vitest';

import App from '@/App.vue';

describe('App', () => {
  /**
   * Purpose: Protects the application shell's primary success message.
   * Description: Mounts the root component and verifies that its rendered text includes the expected heading copy.
   */
  it('renders the application heading', () => {
    const wrapper = mount(App);

    // Expected outcome: The mounted application identifies that startup succeeded.
    // Acceptance criteria: The rendered text contains "You did it!" because that is the root component's user-visible success message.
    expect(wrapper.text()).toContain('You did it!');
  });
});
