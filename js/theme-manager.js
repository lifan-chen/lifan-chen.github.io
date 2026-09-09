(function() {
  'use strict';

  var registry = window.__BLOG_THEME_REGISTRY__;
  if (!registry) return;

  var root = document.documentElement;
  var trigger = document.getElementById('blog-theme-trigger');
  var menu = document.getElementById('blog-theme-menu');
  var choices = menu ? Array.prototype.slice.call(menu.querySelectorAll('[data-theme-id]')) : [];
  var themes = {};
  var aliases = {};

  registry.themes.forEach(function(theme) {
    themes[theme.id] = theme;
    theme.aliases.forEach(function(alias) { aliases[alias] = theme.id; });
  });

  function normalize(id) {
    var candidate = aliases[id] || id;
    return themes[candidate] ? candidate : registry.default;
  }

  function readStored(value) {
    if (!value) return registry.default;
    try {
      var saved = JSON.parse(value);
      if (typeof saved === 'string') return normalize(saved);
      if (saved && (saved.v === registry.version || saved.v == null)) return normalize(saved.id);
    } catch (_) {}
    return registry.default;
  }

  function apply(id, options) {
    options = options || {};
    var previous = root.getAttribute('data-blog-theme') || registry.default;
    var current = normalize(id);

    root.setAttribute('data-blog-theme', current);
    window.__BLOG_THEME_ID__ = current;
    choices.forEach(function(choice) {
      choice.setAttribute('aria-checked', String(choice.getAttribute('data-theme-id') === current));
    });

    if (options.persist) {
      try {
        localStorage.setItem(registry.storageKey, JSON.stringify({
          v: registry.version,
          id: current
        }));
      } catch (_) {}
    }

    if (previous !== current) {
      window.dispatchEvent(new CustomEvent('shimmer:themechange', {
        detail: {
          previous: previous,
          current: current,
          source: options.source || 'api'
        }
      }));
    }
    return current;
  }

  window.ShimmerThemes = {
    list: function() { return registry.themes.slice(); },
    get: function() { return root.getAttribute('data-blog-theme') || registry.default; },
    set: function(id) { return apply(id, { persist: true, source: 'api' }); },
    reset: function() {
      try { localStorage.removeItem(registry.storageKey); } catch (_) {}
      return apply(registry.default, { source: 'api' });
    }
  };

  function setMenu(open, focusIndex) {
    if (!trigger || !menu) return;
    menu.hidden = !open;
    trigger.setAttribute('aria-expanded', String(open));
    if (open && choices.length) choices[focusIndex == null ? 0 : focusIndex].focus();
  }

  if (trigger && menu) {
    apply(window.__BLOG_THEME_ID__);
    trigger.addEventListener('click', function() { setMenu(menu.hidden); });
    trigger.addEventListener('keydown', function(event) {
      if (event.key === 'ArrowDown' || event.key === 'Enter' || event.key === ' ') {
        event.preventDefault(); setMenu(true, 0);
      }
    });
    choices.forEach(function(choice, index) {
      choice.addEventListener('click', function() {
        apply(choice.getAttribute('data-theme-id'), { persist: true, source: 'user' });
        setMenu(false); trigger.focus();
      });
      choice.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') { event.preventDefault(); setMenu(false); trigger.focus(); return; }
        var next = event.key === 'ArrowDown' ? index + 1 : event.key === 'ArrowUp' ? index - 1 : event.key === 'Home' ? 0 : event.key === 'End' ? choices.length - 1 : null;
        if (next != null) { event.preventDefault(); choices[(next + choices.length) % choices.length].focus(); }
      });
    });
    document.addEventListener('pointerdown', function(event) {
      if (!menu.hidden && !event.target.closest('.blog-theme-selector')) setMenu(false);
    });
    document.addEventListener('keydown', function(event) {
      if (event.key === 'Escape' && !menu.hidden) { setMenu(false); trigger.focus(); }
    });
  }

  window.addEventListener('storage', function(event) {
    if (event.key !== registry.storageKey) return;
    apply(readStored(event.newValue), { source: 'storage' });
  });
})();
