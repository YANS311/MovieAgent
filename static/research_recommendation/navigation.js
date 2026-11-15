/* Progressive mobile navigation; independent of scoring and identity logic. */
(function () {
  'use strict';
  const button = document.querySelector('.menu-toggle');
  const menu = document.getElementById('site-menu');
  if (!button || !menu) return;
  document.body.classList.add('navigation-ready');
  function close() {
    menu.classList.remove('is-open');
    button.setAttribute('aria-expanded', 'false');
  }
  button.addEventListener('click', function () {
    const open = button.getAttribute('aria-expanded') !== 'true';
    button.setAttribute('aria-expanded', String(open));
    menu.classList.toggle('is-open', open);
  });
  menu.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') { close(); button.focus(); }
  });
  window.matchMedia('(min-width: 781px)').addEventListener('change', close);
}());
