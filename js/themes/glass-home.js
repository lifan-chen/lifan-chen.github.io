(function() {
  'use strict';

  var root = document.documentElement;
  var hero = document.querySelector('[data-theme-slot="hero"]');
  if (!hero) return;

  var description = hero.querySelector('.theme-hero-description');
  var down = hero.querySelector('.theme-hero-scroll');
  var spacer = document.getElementById('home-spacer');
  var modules = Array.prototype.slice.call(document.querySelectorAll(
    '#about[data-theme-slot="profile"], [data-theme-slot="contribution"], #writing [data-theme-slot="post-card"]'
  ));
  var typingTimer = 0;
  var cycleTimer = 0;
  var active = false;

  function reducedMotion() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function resizeHero() {
    hero.style.height = window.innerHeight + 'px';
  }

  function updateScrollEffects() {
    var scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
    if (reducedMotion()) return;
    modules.forEach(function(module) {
      var top = module.getBoundingClientRect().top;
      module.classList.toggle('glass-bounce-in', scrollTop > top - module.offsetHeight);
    });
  }

  function typeOnce() {
    window.clearTimeout(typingTimer);
    var text = description.getAttribute('data-text') || '';
    var index = 0;
    function step() {
      if (!active) return;
      if (index <= text.length) {
        description.textContent = text.slice(0, index++) + ' | ';
        typingTimer = window.setTimeout(step, 50);
      }
    }
    step();
  }

  function enable() {
    if (active) return;
    active = true;
    resizeHero();
    updateScrollEffects();
    if (reducedMotion()) {
      description.textContent = description.getAttribute('data-text') || '';
    } else {
      typeOnce();
      cycleTimer = window.setInterval(typeOnce, 3000);
    }
    window.addEventListener('resize', resizeHero);
    window.addEventListener('scroll', updateScrollEffects, { passive: true });
  }

  function disable() {
    if (!active) return;
    active = false;
    window.clearTimeout(typingTimer);
    window.clearInterval(cycleTimer);
    description.textContent = description.getAttribute('data-text') || '';
    modules.forEach(function(module) { module.classList.remove('glass-bounce-in'); });
    window.removeEventListener('resize', resizeHero);
    window.removeEventListener('scroll', updateScrollEffects);
  }

  down.addEventListener('click', function(event) {
    event.preventDefault();
    spacer.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth' });
  });

  window.addEventListener('shimmer:themechange', function(event) {
    if (event.detail.current === 'glass') enable(); else disable();
  });

  if (root.getAttribute('data-blog-theme') === 'glass') enable();
})();
