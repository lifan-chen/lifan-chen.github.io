(function () {
  'use strict';

  var element = document.getElementById('double-exposure-config');
  if (!element) return;
  var state;
  try { state = JSON.parse(element.textContent); } catch (_) { return; }
  window.__DOUBLE_EXPOSURE__ = state;

  var registry = state.registry || {};
  var themes = {};
  var aliases = {};
  (registry.themes || []).forEach(function (theme) {
    themes[theme.id] = theme;
    (theme.aliases || []).forEach(function (alias) { aliases[alias] = theme.id; });
  });

  function normalize(id) {
    var candidate = aliases[id] || id;
    return themes[candidate] ? candidate : null;
  }

  function parseSaved(raw) {
    if (!raw) return null;
    try {
      var saved = JSON.parse(raw);
      return normalize(typeof saved === 'string' ? saved : saved && saved.id);
    } catch (_) {
      return normalize(raw);
    }
  }

  function readCookie(name) {
    var prefix = encodeURIComponent(name) + '=';
    var cookies = document.cookie ? document.cookie.split('; ') : [];
    for (var index = 0; index < cookies.length; index += 1) {
      if (cookies[index].indexOf(prefix) === 0) {
        try { return decodeURIComponent(cookies[index].slice(prefix.length)); } catch (_) { return null; }
      }
    }
    return null;
  }

  var selected = null;
  try { selected = parseSaved(localStorage.getItem(registry.storageKey)); } catch (_) {}
  if (!selected) selected = parseSaved(readCookie(registry.storageKey));
  if (!selected) selected = normalize(registry.default) || registry.current;

  var root = document.documentElement;
  root.classList.remove('no-js');
  root.classList.add('js');
  root.setAttribute('data-de-theme', registry.current);
  root.setAttribute('data-blog-theme', registry.current);

  if (selected !== registry.current && themes[selected] && themes[registry.current]) {
    var currentRoot = themes[registry.current].root || '/';
    var targetRoot = themes[selected].root || '/';
    var logicalPath = location.pathname;
    if (currentRoot !== '/' && logicalPath.indexOf(currentRoot) === 0) {
      logicalPath = '/' + logicalPath.slice(currentRoot.length);
    }
    var destination = targetRoot.replace(/\/$/, '') + '/' + logicalPath.replace(/^\//, '') +
      location.search + location.hash;
    location.replace(destination);
  }
})();
