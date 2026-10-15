'use strict';
(() => {
  const gallery = document.getElementById('motion-gallery');
  if (!gallery) return;
  const buttons = gallery.querySelectorAll('[data-motion-example]');
  const panels = gallery.querySelectorAll('[data-motion-panel]');
  for (const button of buttons) {
    button.addEventListener('click', () => {
      for (const choice of buttons) choice.setAttribute('aria-pressed', String(choice === button));
      for (const panel of panels) panel.hidden = panel.dataset.motionPanel !== button.dataset.motionExample;
    });
  }
})();
