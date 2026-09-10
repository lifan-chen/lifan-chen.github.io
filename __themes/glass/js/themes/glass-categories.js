(function() {
  'use strict';

  var root = document.documentElement;
  var container = document.getElementById('categories');
  var indexNode = document.getElementById('category-post-index');
  var results = container && container.querySelector('.category-featured-list');
  if (!container || !indexNode || !results) return;

  var categories;
  try { categories = JSON.parse(indexNode.textContent); } catch (_) { return; }
  var byPath = {};
  categories.forEach(function(category) { byPath[new URL(category.path, location.href).pathname] = category; });
  var links = Array.prototype.slice.call(container.querySelectorAll('.category-list a[href]'));
  var originalResults = results.innerHTML;
  var initialized = false;
  var pageSize = 10;
  var currentPath = '';
  var currentPage = 1;

  function totalPages(category) {
    return Math.max(1, Math.ceil(category.posts.length / pageSize));
  }

  function pageButton(label, page, current, ariaLabel) {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'category-pagination-button';
    button.textContent = label;
    button.dataset.page = String(page);
    if (ariaLabel) button.setAttribute('aria-label', ariaLabel);
    if (page === current) {
      button.classList.add('is-current');
      button.setAttribute('aria-current', 'page');
    }
    return button;
  }

  function renderPagination(category, page) {
    var count = totalPages(category);
    if (count <= 1) return;
    var nav = document.createElement('nav');
    nav.className = 'category-pagination';
    nav.setAttribute('aria-label', 'Category posts pagination');
    var previous = pageButton('\u2039', page - 1, page, 'Previous page');
    previous.disabled = page === 1;
    nav.appendChild(previous);
    for (var number = 1; number <= count; number += 1) {
      nav.appendChild(pageButton(String(number), number, page, 'Page ' + number));
    }
    var next = pageButton('\u203a', page + 1, page, 'Next page');
    next.disabled = page === count;
    nav.appendChild(next);
    nav.addEventListener('click', function(event) {
      var target = event.target.closest('button[data-page]');
      if (!target || target.disabled) return;
      render(currentPath, Number(target.dataset.page), true, false);
    });
    results.appendChild(nav);
  }

  function render(path, requestedPage, updateHistory, replaceHistory) {
    var category = byPath[path];
    if (!category) return false;
    var page = Math.min(Math.max(Number(requestedPage) || 1, 1), totalPages(category));
    currentPath = path;
    currentPage = page;
    results.replaceChildren();
    var heading = document.createElement('h2');
    heading.className = 'categoryName';
    heading.textContent = category.name;
    results.appendChild(heading);
    var list = document.createElement('ol');
    list.className = 'category-filter-posts';
    var start = (page - 1) * pageSize;
    category.posts.slice(start, start + pageSize).forEach(function(post) {
      var item = document.createElement('li');
      var anchor = document.createElement('a');
      anchor.href = post.path;
      anchor.textContent = post.title;
      item.appendChild(anchor);
      list.appendChild(item);
    });
    if (!category.posts.length) {
      var empty = document.createElement('li');
      empty.className = 'category-filter-empty';
      empty.textContent = 'No posts in this category.';
      list.appendChild(empty);
    }
    results.appendChild(list);
    renderPagination(category, page);
    links.forEach(function(link) {
      var selected = new URL(link.href, location.href).pathname === path;
      link.classList.toggle('is-active', selected);
      if (selected) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
    });
    if (updateHistory) {
      var query = '?category=' + encodeURIComponent(path) + (page > 1 ? '&page=' + page : '');
      var state = { category: path, page: page };
      if (replaceHistory) history.replaceState(state, '', query);
      else history.pushState(state, '', query);
    }
    initialized = true;
    return true;
  }

  function enable() {
    if (initialized) return;
    var params = new URLSearchParams(location.search);
    var requested = params.get('category');
    var page = Number(params.get('page')) || 1;
    if (requested && render(requested, page, false, false)) return;
    if (links.length) render(new URL(links[0].href, location.href).pathname, 1, false, false);
  }

  function disable() {
    if (!initialized) return;
    results.innerHTML = originalResults;
    initialized = false;
    links.forEach(function(link) {
      link.classList.remove('is-active');
      link.removeAttribute('aria-current');
    });
  }

  links.forEach(function(link) {
    link.addEventListener('click', function(event) {
      if (root.getAttribute('data-blog-theme') !== 'glass') return;
      var path = new URL(link.href, location.href).pathname;
      if (!byPath[path]) return;
      event.preventDefault();
      render(path, 1, true, false);
    });
  });
  window.addEventListener('popstate', function(event) {
    if (event.state && event.state.category) {
      render(event.state.category, event.state.page || 1, false, false);
      return;
    }
    var params = new URLSearchParams(location.search);
    var path = params.get('category');
    if (path) render(path, Number(params.get('page')) || 1, false, false);
  });
  window.addEventListener('shimmer:themechange', function(event) {
    if (event.detail.current === 'glass') enable(); else disable();
  });
  if (root.getAttribute('data-blog-theme') === 'glass') enable();
})();
