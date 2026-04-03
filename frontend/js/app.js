// EarthAI SPA hash router
(() => {
  const PAGES = {
    rainfall:   { title: 'Rainfall Analysis',       icon: 'rainy' },
    optical:    { title: 'Temporal Analysis Canvas', icon: 'visibility' },
    sar:        { title: 'SAR Detection Analytics',  icon: 'radar' },
    classifier: { title: 'AI Flood Classifier',      icon: 'psychology' },
    flappy:     { title: 'Flappy Bird Competition',  icon: 'sports_esports' },
  };

  const DEFAULT_PAGE = 'rainfall';
  const pageCache = {};

  function currentPage() {
    const hash = window.location.hash.replace('#', '');
    return PAGES[hash] ? hash : DEFAULT_PAGE;
  }

  function updateSidebar(page) {
    document.querySelectorAll('.nav-link').forEach(link => {
      const linkPage = link.getAttribute('data-page');
      if (linkPage === page) {
        link.classList.add('bg-white/10', 'text-white');
        link.classList.remove('text-slate-400');
      } else {
        link.classList.remove('bg-white/10', 'text-white');
        link.classList.add('text-slate-400');
      }
    });
  }

  function updateTopbar(page) {
    const titleEl = document.getElementById('topbar-title');
    if (titleEl && PAGES[page]) {
      titleEl.textContent = PAGES[page].title;
    }
  }

  async function loadPage(page) {
    const content = document.getElementById('page-content');
    if (!content) return;

    // Show loading state
    if (!pageCache[page]) {
      content.innerHTML = `
        <div class="flex items-center justify-center py-20">
          <div class="text-center">
            <span class="material-symbols-outlined mb-3 text-5xl text-outline animate-pulse">hourglass_top</span>
            <p class="text-sm text-on-surface-variant">Loading ${PAGES[page]?.title || page}...</p>
          </div>
        </div>`;
    }

    try {
      if (!pageCache[page]) {
        const resp = await fetch(`/pages/${page}.html`);
        if (!resp.ok) throw new Error(`Page not found: ${page}`);
        pageCache[page] = await resp.text();
      }
      content.innerHTML = pageCache[page];

      // Call page init function if it exists
      const initFn = window[`init_${page}`];
      if (typeof initFn === 'function') {
        initFn();
      }
    } catch (err) {
      content.innerHTML = `
        <div class="flex items-center justify-center py-20">
          <div class="text-center">
            <span class="material-symbols-outlined mb-3 text-5xl text-error">error</span>
            <p class="text-sm text-on-surface-variant">Failed to load page: ${page}</p>
          </div>
        </div>`;
      console.error('Page load error:', err);
    }
  }

  function navigate(page) {
    updateSidebar(page);
    updateTopbar(page);
    loadPage(page);
  }

  // Listen for hash changes
  window.addEventListener('hashchange', () => {
    navigate(currentPage());
  });

  // Initial load
  if (!window.location.hash) {
    window.location.hash = `#${DEFAULT_PAGE}`;
  } else {
    navigate(currentPage());
  }
})();
