(function() {
  'use strict';

  var registry = window.__BLOG_THEME_REGISTRY__;
  if (!registry) return;

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
    if (!value) return registry.current;
    try {
      var saved = JSON.parse(value);
      if (typeof saved === 'string') return normalize(saved);
      if (saved && (saved.v === registry.version || saved.v == null)) return normalize(saved.id);
    } catch (_) {}
    return registry.current;
  }

  function destination(id) {
    var currentTheme = themes[registry.current];
    var targetTheme = themes[normalize(id)];
    if (!currentTheme || !targetTheme) return null;
    var logicalPath = location.pathname;
    if (currentTheme.root !== '/' && logicalPath.indexOf(currentTheme.root) === 0) {
      logicalPath = '/' + logicalPath.slice(currentTheme.root.length);
    }
    return targetTheme.root.replace(/\/$/, '') + '/' + logicalPath.replace(/^\//, '') +
      location.search + location.hash;
  }

  function select(id, replace) {
    var selected = normalize(id);
    try {
      localStorage.setItem(registry.storageKey, JSON.stringify({ v: registry.version, id: selected }));
    } catch (_) {}
    if (selected === registry.current) return selected;
    var target = destination(selected);
    if (target) {
      if (replace) location.replace(target); else location.assign(target);
    }
    return selected;
  }

  function updateChoices() {
    choices.forEach(function(choice) {
      choice.setAttribute('aria-checked', String(choice.getAttribute('data-theme-id') === registry.current));
    });
  }

  window.DoubleExposureThemes = {
    list: function() { return registry.themes.slice(); },
    get: function() { return registry.current; },
    set: function(id) { return select(id, false); },
    reset: function() {
      try { localStorage.removeItem(registry.storageKey); } catch (_) {}
      return select(registry.default, false);
    }
  };

  function setMenu(open, focusIndex) {
    if (!trigger || !menu) return;
    menu.hidden = !open;
    trigger.setAttribute('aria-expanded', String(open));
    if (open && choices.length) choices[focusIndex == null ? 0 : focusIndex].focus();
  }

  if (trigger && menu) {
    updateChoices();
    trigger.addEventListener('click', function() { setMenu(menu.hidden); });
    trigger.addEventListener('keydown', function(event) {
      if (event.key === 'ArrowDown' || event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        setMenu(true, 0);
      }
    });
    choices.forEach(function(choice, index) {
      choice.addEventListener('click', function() {
        select(choice.getAttribute('data-theme-id'), false);
      });
      choice.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
          event.preventDefault();
          setMenu(false);
          trigger.focus();
          return;
        }
        var next = event.key === 'ArrowDown' ? index + 1 :
          event.key === 'ArrowUp' ? index - 1 :
          event.key === 'Home' ? 0 :
          event.key === 'End' ? choices.length - 1 : null;
        if (next != null) {
          event.preventDefault();
          choices[(next + choices.length) % choices.length].focus();
        }
      });
    });
    document.addEventListener('pointerdown', function(event) {
      if (!menu.hidden && !event.target.closest('.blog-theme-selector')) setMenu(false);
    });
    document.addEventListener('keydown', function(event) {
      if (event.key === 'Escape' && !menu.hidden) {
        setMenu(false);
        trigger.focus();
      }
    });
  }

  window.addEventListener('storage', function(event) {
    if (event.key !== registry.storageKey) return;
    var selected = readStored(event.newValue);
    if (selected !== registry.current) select(selected, true);
  });
})();
