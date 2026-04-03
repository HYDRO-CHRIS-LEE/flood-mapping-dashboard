/* Optical / Temporal Analysis Canvas page logic */

function init_optical() {
  const selector = document.getElementById('event-select');
  const mapDiv   = document.getElementById('optical-map');

  if (!selector || !mapDiv) return;

  // ── State ───────────────────────────────────────────────────
  let map = null;
  let imageOverlay = null;
  let currentEvent = null;
  let currentLayer = 'RGB';
  let currentPeriod = 'before';
  let boundsData = null;

  // ── Populate event dropdown ─────────────────────────────────
  API.get('/events').then(events => {
    events.forEach(ev => {
      const opt = document.createElement('option');
      opt.value = ev.key;
      opt.textContent = `${ev.label} (${ev.year})`;
      selector.appendChild(opt);
    });
    if (events.length > 0) {
      selector.value = events[0].key;
      loadEvent(events[0].key);
    }
  }).catch(err => {
    console.error('Failed to load events:', err);
  });

  // ── Event handlers ──────────────────────────────────────────
  selector.addEventListener('change', () => {
    if (selector.value) loadEvent(selector.value);
  });

  // Layer radio buttons
  document.querySelectorAll('input[name="optical-layer"]').forEach(radio => {
    radio.addEventListener('change', () => {
      currentLayer = radio.value;
      // Hide period controls for NDWI_change
      const periodGroup = document.getElementById('period-group');
      if (periodGroup) {
        periodGroup.style.display = currentLayer === 'NDWI_change' ? 'none' : '';
      }
      updateOverlay();
    });
  });

  // Period radio buttons
  document.querySelectorAll('input[name="optical-period"]').forEach(radio => {
    radio.addEventListener('change', () => {
      currentPeriod = radio.value;
      updateOverlay();
    });
  });

  // ── Load event bounds + init map ────────────────────────────
  function loadEvent(key) {
    currentEvent = key;

    // Show loading state on map
    mapDiv.innerHTML = `
      <div class="flex items-center justify-center h-full">
        <div class="text-center">
          <span class="material-symbols-outlined mb-3 text-4xl text-outline animate-pulse">hourglass_top</span>
          <p class="text-sm text-on-surface-variant">Loading map...</p>
        </div>
      </div>`;

    // Reset map instance
    if (map) {
      map.remove();
      map = null;
    }
    imageOverlay = null;

    API.get(`/tif/${key}/bounds`).then(data => {
      boundsData = data;
      initMap();
      updateOverlay();
    }).catch(err => {
      mapDiv.innerHTML = `
        <div class="flex items-center justify-center h-full">
          <p class="text-sm text-error">Failed to load map bounds: ${err.message}</p>
        </div>`;
    });
  }

  // ── Initialize Leaflet map ──────────────────────────────────
  function initMap() {
    mapDiv.innerHTML = '';

    map = L.map(mapDiv, {
      center: boundsData.center,
      zoom: boundsData.zoom,
      zoomControl: false,
    });

    // Zoom control at bottom-right
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // CartoDB positron basemap
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(map);
  }

  // ── Update image overlay ────────────────────────────────────
  function updateOverlay() {
    if (!map || !boundsData || !currentEvent) return;

    // For NDWI_change, period is ignored by the backend but we still need a value
    const period = currentLayer === 'NDWI_change' ? 'after' : currentPeriod;
    const url = `/api/tif/${currentEvent}/${currentLayer}/${period}`;

    // Remove existing overlay
    if (imageOverlay) {
      map.removeLayer(imageOverlay);
      imageOverlay = null;
    }

    const bounds = L.latLngBounds(boundsData.bounds);

    imageOverlay = L.imageOverlay(url, bounds, {
      opacity: 0.85,
      interactive: false,
    }).addTo(map);

    // Fit map to image bounds
    map.fitBounds(bounds);
  }
}
