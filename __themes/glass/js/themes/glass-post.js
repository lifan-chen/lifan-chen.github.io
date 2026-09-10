(function() {
  'use strict';

  var root = document.documentElement;
  var layout = document.querySelector('[data-theme-slot="post-layout"]');
  var toc = document.querySelector('[data-theme-slot="post-toc"]');
  if (!layout || !toc) return;

  var links = Array.prototype.slice.call(toc.querySelectorAll('a[href^="#"]'));
  var headings = links.map(function(link) {
    try { return document.getElementById(decodeURIComponent(link.hash.slice(1))); } catch (_) { return null; }
  });
  var active = false;
  var frame = 0;

  function renderActive() {
    frame = 0;
    if (!active) return;
    var current = -1;
    headings.forEach(function(heading, index) {
      if (heading && heading.getBoundingClientRect().top <= 135) current = index;
    });
    links.forEach(function(link, index) {
      var selected = index === current;
      link.classList.toggle('is-active', selected);
      if (selected) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }

  function schedule() {
    if (!frame) frame = window.requestAnimationFrame(renderActive);
  }

  function enable() {
    if (active) return;
    active = true;
    schedule();
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
  }

  function disable() {
    active = false;
    if (frame) window.cancelAnimationFrame(frame);
    frame = 0;
    links.forEach(function(link) { link.classList.remove('is-active'); link.removeAttribute('aria-current'); });
    window.removeEventListener('scroll', schedule);
    window.removeEventListener('resize', schedule);
  }

  window.addEventListener('shimmer:themechange', function(event) {
    if (event.detail.current === 'glass') enable(); else disable();
  });
  if (root.getAttribute('data-blog-theme') === 'glass') enable();
})();
