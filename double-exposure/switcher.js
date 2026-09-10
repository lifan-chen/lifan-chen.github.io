(function () {
  'use strict';

  var input = window.__DOUBLE_EXPOSURE__;
  if (!input || !input.registry) return;

  var registry = input.registry;
  var switcher = input.switcher || {};
  var presentation = switcher.presentation || {};
  var currentThemeId = input.current || registry.current || registry.default;
  var themes = {};
  var aliases = {};
  var instances = [];
  var instanceSequence = 0;

  (Array.isArray(registry.themes) ? registry.themes : []).forEach(function (theme) {
    if (!theme || !theme.id) return;
    themes[theme.id] = theme;
    (Array.isArray(theme.aliases) ? theme.aliases : []).forEach(function (alias) {
      aliases[alias] = theme.id;
    });
  });

  function normalize(id) {
    var candidate = aliases[id] || id;
    return themes[candidate] ? candidate : registry.default;
  }

  function storageKey() {
    return registry.storageKey || registry.storage_key || 'double-exposure.theme.preference';
  }

  function decodeStored(value) {
    if (!value) return null;
    try {
      var parsed = JSON.parse(value);
      if (typeof parsed === 'string') return normalize(parsed);
      if (parsed && typeof parsed.id === 'string') return normalize(parsed.id);
    } catch (_) {
      return normalize(value);
    }
    return null;
  }

  function cookiePath() {
    if (registry.cookiePath || registry.cookie_path) {
      return registry.cookiePath || registry.cookie_path;
    }
    var defaultTheme = themes[registry.default];
    return defaultTheme && defaultTheme.root ? defaultTheme.root : '/';
  }

  function readCookie() {
    var prefix = encodeURIComponent(storageKey()) + '=';
    var cookies = document.cookie ? document.cookie.split('; ') : [];
    for (var index = 0; index < cookies.length; index += 1) {
      if (cookies[index].indexOf(prefix) === 0) {
        try { return decodeURIComponent(cookies[index].slice(prefix.length)); } catch (_) { return null; }
      }
    }
    return null;
  }

  function readPreference() {
    var value = null;
    try { value = window.localStorage.getItem(storageKey()); } catch (_) {}
    return decodeStored(value || readCookie());
  }

  function persist(id) {
    var selected = normalize(id);
    try { window.localStorage.setItem(storageKey(), selected); } catch (_) {}
    try {
      document.cookie = encodeURIComponent(storageKey()) + '=' + encodeURIComponent(selected) +
        '; Path=' + cookiePath() + '; Max-Age=31536000; SameSite=Lax';
    } catch (_) {}
    return selected;
  }

  function clearPreference() {
    try { window.localStorage.removeItem(storageKey()); } catch (_) {}
    try {
      document.cookie = encodeURIComponent(storageKey()) + '=; Path=' + cookiePath() +
        '; Max-Age=0; SameSite=Lax';
    } catch (_) {}
  }

  function normalizedRoot(root) {
    var value = typeof root === 'string' && root ? root : '/';
    if (value.charAt(0) !== '/') value = '/' + value;
    return value === '/' ? '/' : value.replace(/\/+$/, '') + '/';
  }

  function logicalPath() {
    var current = themes[currentThemeId];
    var root = normalizedRoot(current && current.root);
    var path = window.location.pathname || '/';
    if (root !== '/') {
      var rootWithoutSlash = root.replace(/\/$/, '');
      if (path === rootWithoutSlash) return '/';
      if (path.indexOf(root) === 0) path = '/' + path.slice(root.length);
    }
    return path.charAt(0) === '/' ? path : '/' + path;
  }

  function destination(id) {
    var target = themes[normalize(id)];
    if (!target) return null;
    var root = normalizedRoot(target.root);
    var path = logicalPath().replace(/^\/+/, '');
    return (root + path).replace(/\/{2,}/g, '/') + window.location.search + window.location.hash;
  }

  function select(id, options) {
    options = options || {};
    var selected = normalize(id);
    if (!themes[selected]) return null;
    if (options.persist !== false) persist(selected);
    if (selected === currentThemeId) {
      closeMenus();
      return selected;
    }
    var target = destination(selected);
    if (target) {
      if (options.replace) window.location.replace(target);
      else window.location.assign(target);
    }
    return selected;
  }

  function selectorIsPresent(value) {
    var selectors = Array.isArray(value) ? value : [value];
    return selectors.some(function (selector) {
      if (typeof selector !== 'string') return false;
      try { return !!document.querySelector(selector); } catch (_) { return false; }
    });
  }

  function mountIsEnabled(mount) {
    if (!mount || mount.enabled === false) return false;
    var whenPresent = mount.whenPresent || mount.when_present;
    var unlessPresent = mount.unlessPresent || mount.unless_present;
    if (whenPresent && !selectorIsPresent(whenPresent)) return false;
    if (unlessPresent && selectorIsPresent(unlessPresent)) return false;
    if (mount.media && window.matchMedia && !window.matchMedia(mount.media).matches) return false;
    if (mount.path) {
      try {
        var expression = mount.path instanceof RegExp ? mount.path : new RegExp(mount.path);
        if (!expression.test(logicalPath())) return false;
      } catch (_) { return false; }
    }
    return true;
  }

  function paletteIcon() {
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('focusable', 'false');
    svg.setAttribute('aria-hidden', 'true');
    var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M12 3a9 9 0 1 0 0 18h1.1a1.9 1.9 0 0 0 1.2-3.4 1.5 1.5 0 0 1 1-2.6H17a4 4 0 0 0 4-4c0-4.4-4-8-9-8Zm-5 9.2a1.2 1.2 0 1 1 0-2.4 1.2 1.2 0 0 1 0 2.4Zm2-4a1.2 1.2 0 1 1 0-2.4 1.2 1.2 0 0 1 0 2.4Zm4-.8A1.2 1.2 0 1 1 13 5a1.2 1.2 0 0 1 0 2.4Zm4 2a1.2 1.2 0 1 1 0-2.4 1.2 1.2 0 0 1 0 2.4Z');
    svg.appendChild(path);
    return svg;
  }

  function createIcon() {
    if (presentation.icon === false || presentation.icon === 'none') return null;
    var holder = document.createElement('span');
    holder.className = 'blog-theme-icon double-exposure-switcher__icon de-theme-icon';
    holder.setAttribute('aria-hidden', 'true');
    if (presentation.icon && typeof presentation.icon === 'object') {
      if (presentation.icon.className) holder.className += ' ' + presentation.icon.className;
      if (presentation.icon.text) holder.textContent = presentation.icon.text;
    } else if (presentation.icon && presentation.icon !== 'palette') {
      holder.textContent = presentation.icon;
    } else {
      holder.appendChild(paletteIcon());
    }
    return holder;
  }

  function closeInstance(instance, restoreFocus) {
    if (!instance || instance.menu.hidden) return;
    instance.menu.hidden = true;
    instance.trigger.setAttribute('aria-expanded', 'false');
    if (restoreFocus) instance.trigger.focus();
  }

  function closeMenus(except) {
    instances.forEach(function (instance) {
      if (instance !== except) closeInstance(instance, false);
    });
  }

  function openInstance(instance, focusIndex) {
    closeMenus(instance);
    instance.menu.hidden = false;
    instance.trigger.setAttribute('aria-expanded', 'true');
    var choices = instance.choices;
    if (!choices.length) return;
    var index = typeof focusIndex === 'number' ? focusIndex : choices.findIndex(function (choice) {
      return choice.getAttribute('aria-checked') === 'true';
    });
    choices[index < 0 ? 0 : index].focus();
  }

  function createInstance(mount, target) {
    instanceSequence += 1;
    var tag = /^(li|div|span)$/.test(mount.wrapper || '') ? mount.wrapper : 'div';
    var root = document.createElement(tag);
    var mountId = mount.id || 'mount-' + instanceSequence;
    root.className = 'blog-theme-selector double-exposure-switcher de-theme-switcher';
    if (mount.className || mount.class) root.className += ' ' + (mount.className || mount.class);
    root.setAttribute('data-theme-slot', 'theme-selector');
    root.setAttribute('data-double-exposure-switcher', '');
    root.setAttribute('data-de-mount', mountId);

    var trigger = document.createElement('button');
    var menu = document.createElement(tag === 'span' ? 'span' : 'div');
    var menuId = 'double-exposure-theme-menu-' + instanceSequence;
    trigger.type = 'button';
    trigger.className = 'blog-theme-trigger double-exposure-switcher__trigger de-theme-trigger';
    trigger.setAttribute('aria-haspopup', 'menu');
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-controls', menuId);
    trigger.setAttribute('aria-label', presentation.aria_label || presentation.ariaLabel || 'Choose visual theme');
    var icon = createIcon();
    if (icon) trigger.appendChild(icon);
    var label = document.createElement('span');
    label.className = 'blog-theme-label double-exposure-switcher__label';
    label.textContent = presentation.label || 'Theme';
    trigger.appendChild(label);

    menu.id = menuId;
    menu.className = 'blog-theme-menu double-exposure-switcher__menu de-theme-menu';
    menu.setAttribute('role', 'menu');
    menu.hidden = true;
    var choices = [];
    (registry.themes || []).forEach(function (theme) {
      if (!theme || !theme.id) return;
      var choice = document.createElement('button');
      choice.type = 'button';
      choice.className = 'double-exposure-switcher__choice de-theme-option';
      choice.setAttribute('role', 'menuitemradio');
      choice.setAttribute('data-theme-id', theme.id);
      choice.setAttribute('aria-checked', String(theme.id === currentThemeId));
      choice.tabIndex = -1;
      choice.textContent = theme.label || theme.id;
      menu.appendChild(choice);
      choices.push(choice);
    });
    root.appendChild(trigger);
    root.appendChild(menu);

    var instance = {
      root: root,
      trigger: trigger,
      menu: menu,
      choices: choices,
      mountId: mountId,
      target: target
    };
    trigger.addEventListener('click', function () {
      if (menu.hidden) openInstance(instance); else closeInstance(instance, false);
    });
    trigger.addEventListener('keydown', function (event) {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openInstance(instance, event.key === 'ArrowUp' ? choices.length - 1 : undefined);
      }
    });
    choices.forEach(function (choice, index) {
      choice.addEventListener('click', function () { select(choice.getAttribute('data-theme-id')); });
      choice.addEventListener('keydown', function (event) {
        var next = null;
        if (event.key === 'Escape') {
          event.preventDefault();
          closeInstance(instance, true);
          return;
        }
        if (event.key === 'ArrowDown') next = index + 1;
        else if (event.key === 'ArrowUp') next = index - 1;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = choices.length - 1;
        else if (event.key === 'Tab') closeInstance(instance, false);
        if (next !== null && choices.length) {
          event.preventDefault();
          choices[(next + choices.length) % choices.length].focus();
        }
      });
    });
    instances.push(instance);

    var position = mount.insert || 'append';
    if (position === 'prepend') target.insertBefore(root, target.firstChild);
    else if (position === 'before') target.parentNode.insertBefore(root, target);
    else if (position === 'after') target.parentNode.insertBefore(root, target.nextSibling);
    else target.appendChild(root);
    return instance;
  }

  function mountAll() {
    (Array.isArray(switcher.mounts) ? switcher.mounts : []).forEach(function (mount) {
      if (!mountIsEnabled(mount) || !mount.target) return;
      var targets;
      try { targets = document.querySelectorAll(mount.target); } catch (_) { targets = []; }
      if (!targets.length) {
        if (mount.required && window.console) console.error('[Double Exposure] Missing switcher mount target:', mount.target);
        return;
      }
      Array.prototype.slice.call(targets, 0, mount.all ? targets.length : 1).forEach(function (target) {
        var mountId = mount.id || mount.target;
        var alreadyMounted = instances.some(function (instance) {
          return instance.mountId === mountId && instance.target === target && instance.root.isConnected;
        });
        if (alreadyMounted) return;
        createInstance(mount, target);
      });
    });
    return instances.slice();
  }

  window.DoubleExposureThemes = {
    list: function () { return (registry.themes || []).slice(); },
    get: function () { return currentThemeId; },
    getPreference: readPreference,
    destination: destination,
    set: function (id) { return select(id); },
    reset: function () {
      clearPreference();
      return select(registry.default, { persist: false });
    },
    refresh: mountAll,
    close: closeMenus
  };

  document.addEventListener('pointerdown', function (event) {
    if (!event.target.closest || !event.target.closest('[data-double-exposure-switcher]')) closeMenus();
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') closeMenus();
  });
  window.addEventListener('storage', function (event) {
    if (event.key !== storageKey()) return;
    var selected = decodeStored(event.newValue || readCookie());
    if (selected && selected !== currentThemeId) select(selected, { replace: true, persist: false });
  });

  document.documentElement.setAttribute('data-de-theme', currentThemeId);
  document.documentElement.setAttribute('data-blog-theme', currentThemeId);

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountAll);
  else mountAll();
})();
