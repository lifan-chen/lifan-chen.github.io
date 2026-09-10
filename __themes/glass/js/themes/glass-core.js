(function() {
  'use strict';

  var root = document.documentElement;
  var header = document.getElementById('header');
  if (!header) return;
  var active = false;
  var queued = false;

  function renderHeaderState() {
    queued = false;
    if (!active) {
      header.classList.remove('is-scrolled');
      return;
    }
    header.classList.toggle('is-scrolled', window.scrollY > 0);
  }

  function scheduleHeaderState() {
    if (queued) return;
    queued = true;
    window.requestAnimationFrame(renderHeaderState);
  }

  function enable() {
    if (active) return;
    active = true;
    renderHeaderState();
    window.addEventListener('scroll', scheduleHeaderState, { passive: true });
  }

  function disable() {
    if (!active) return;
    active = false;
    queued = false;
    window.removeEventListener('scroll', scheduleHeaderState);
    header.classList.remove('is-scrolled');
  }

  window.addEventListener('shimmer:themechange', function(event) {
    if (event.detail.current === 'glass') enable(); else disable();
  });

  if (root.getAttribute('data-blog-theme') === 'glass') enable();
})();
