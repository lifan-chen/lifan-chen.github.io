(function() {
  var archive = document.querySelector('#archive.archive-infinite');
  if (!archive) return;

  var list = archive.querySelector('.post-list');
  var pagination = archive.querySelector('.pagination');
  var status = archive.querySelector('.archive-load-status');
  if (!list || !pagination) return;

  archive.classList.add('infinite-ready');

  var nextLink = pagination.querySelector('.pagination-arrow[aria-label="Next page"]');
  var nextUrl = nextLink ? nextLink.getAttribute('href') : '';
  var loading = false;
  var done = !nextUrl;
  var observer = null;

  function setStatus(message) {
    if (status) status.textContent = message || '';
  }

  function lastYear() {
    var years = list.querySelectorAll('.post-year h2');
    return years.length ? years[years.length - 1].textContent.trim() : '';
  }

  function appendNextPage(doc) {
    var nextArchive = doc.querySelector('#archive');
    if (!nextArchive) return '';

    var incoming = nextArchive.querySelectorAll('.post-list > li');
    var currentLastYear = lastYear();
    var fragment = document.createDocumentFragment();

    incoming.forEach(function(item, index) {
      if (
        index === 0 &&
        item.classList.contains('post-year') &&
        item.textContent.trim() === currentLastYear
      ) {
        return;
      }
      fragment.appendChild(document.importNode(item, true));
    });

    list.appendChild(fragment);

    var nextPageLink = nextArchive.querySelector('.pagination-arrow[aria-label="Next page"]');
    return nextPageLink ? nextPageLink.getAttribute('href') : '';
  }

  function loadNext() {
    if (loading || done) return;
    loading = true;
    archive.classList.add('is-loading');
    setStatus('Loading more...');

    fetch(nextUrl, { credentials: 'same-origin' })
      .then(function(response) {
        if (!response.ok) throw new Error('Failed to load archive page');
        return response.text();
      })
      .then(function(html) {
        var doc = new DOMParser().parseFromString(html, 'text/html');
        nextUrl = appendNextPage(doc);
        done = !nextUrl;
        setStatus(done ? 'All posts loaded.' : '');
      })
      .catch(function() {
        done = true;
        archive.classList.add('load-error');
        setStatus('Could not load more posts. Please refresh and try again.');
      })
      .finally(function() {
        loading = false;
        archive.classList.remove('is-loading');
        if (done && observer) observer.disconnect();
      });
  }

  if (!done && 'IntersectionObserver' in window) {
    observer = new IntersectionObserver(function(entries) {
      if (entries.some(function(entry) { return entry.isIntersecting; })) {
        loadNext();
      }
    }, {
      rootMargin: '400px 0px'
    });
    observer.observe(status || pagination);
  } else if (!done) {
    window.addEventListener('scroll', function() {
      var distanceToBottom = document.documentElement.scrollHeight - window.innerHeight - window.scrollY;
      if (distanceToBottom < 500) loadNext();
    });
  }
})();
