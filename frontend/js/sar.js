/* SAR Detection Analytics page logic */

function init_sar() {
  const selector   = document.getElementById('sar-event-select');
  const mapDiv     = document.getElementById('sar-map');
  const liveBtn    = document.getElementById('sar-live-btn');
  const sliderEl   = document.getElementById('sar-threshold');
  const sliderVal  = document.getElementById('sar-threshold-val');
  const otsuBtn    = document.getElementById('sar-otsu-btn');
  const permCb     = document.getElementById('sar-perm-water');

  if (!selector || !mapDiv) return;

  // ── State ───────────────────────────────────────────────────
  let map = null;
  let imageOverlay = null;
  let currentEvent = null;
  let otsuVal = -16.0;
  let loading = false;

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
      currentEvent = events[0].key;
      computeSAR();
    }
  }).catch(err => {
    console.error('Failed to load events:', err);
  });

  // ── Event handlers ──────────────────────────────────────────
  selector.addEventListener('change', () => {
    if (selector.value) {
      currentEvent = selector.value;
      computeSAR();
    }
  });

  if (liveBtn) {
    liveBtn.addEventListener('click', () => {
      if (currentEvent) computeSAR();
    });
  }

  // Threshold slider change
  sliderEl.addEventListener('input', () => {
    sliderVal.textContent = parseFloat(sliderEl.value).toFixed(1) + ' dB';
  });

  sliderEl.addEventListener('change', () => {
    computeSAR();
  });

  // Otsu auto-threshold button
  otsuBtn.addEventListener('click', () => {
    sliderEl.value = otsuVal;
    sliderVal.textContent = otsuVal.toFixed(1) + ' dB';
    computeSAR();
  });

  // Permanent water checkbox
  permCb.addEventListener('change', () => {
    computeSAR();
  });

  // ── Compute SAR ─────────────────────────────────────────────
  function computeSAR() {
    if (!currentEvent || loading) return;
    loading = true;

    const threshold = parseFloat(sliderEl.value);
    const removePermanent = permCb.checked;

    API.post('/sar/compute', {
      event: currentEvent,
      threshold: threshold,
      remove_permanent: removePermanent,
    }).then(data => {
      loading = false;
      otsuVal = data.otsu;

      // If this is the first call (no overlay yet), set slider to Otsu
      if (!imageOverlay && data.otsu) {
        sliderEl.value = data.otsu;
        sliderVal.textContent = data.otsu.toFixed(1) + ' dB';
      }

      updateMetrics(data.metrics, data.otsu);
      updateMap(data);
      updateHistogram(data.histogram, parseFloat(sliderEl.value), data.otsu);
    }).catch(err => {
      loading = false;
      console.error('SAR compute error:', err);
    });
  }

  // ── Update metrics panel ────────────────────────────────────
  function updateMetrics(m, otsu) {
    const km2El = document.getElementById('sar-metric-km2');
    const pctEl = document.getElementById('sar-metric-pct');
    const pxEl  = document.getElementById('sar-metric-px');
    const otsuEl = document.getElementById('sar-metric-otsu');

    if (km2El) km2El.innerHTML = m.flood_km2.toFixed(2) + '<span class="text-base font-bold ml-1">km\u00B2</span>';
    if (pctEl) pctEl.innerHTML = m.flood_pct.toFixed(2) + '<span class="text-base font-bold ml-1">%</span>';
    if (pxEl)  pxEl.textContent = m.flood_px.toLocaleString();
    if (otsuEl) otsuEl.innerHTML = otsu.toFixed(1) + '<span class="text-base font-bold ml-1">dB</span>';
  }

  // ── Update Leaflet map ──────────────────────────────────────
  function updateMap(data) {
    if (!data.bounds) return;

    const bounds = L.latLngBounds(data.bounds);

    if (!map) {
      mapDiv.innerHTML = '';
      map = L.map(mapDiv, {
        center: data.center || bounds.getCenter(),
        zoom: data.zoom || 9,
        zoomControl: false,
      });
      L.control.zoom({ position: 'bottomright' }).addTo(map);

      // CartoDB dark_matter basemap
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19,
      }).addTo(map);
    }

    // Remove existing overlay
    if (imageOverlay) {
      map.removeLayer(imageOverlay);
      imageOverlay = null;
    }

    // Add new flood overlay
    imageOverlay = L.imageOverlay(data.tile_url, bounds, {
      opacity: 0.85,
      interactive: false,
    }).addTo(map);

    map.fitBounds(bounds);
  }

  // ── Update Plotly histogram ─────────────────────────────────
  function updateHistogram(hist, threshold, otsu) {
    const colors = hist.centers.map(c => c < threshold ? '#2563eb' : '#94a3b8');

    const barTrace = {
      x: hist.centers,
      y: hist.counts,
      type: 'bar',
      marker: { color: colors },
      name: 'Pixel count',
      hovertemplate: '%{x:.1f} dB<br>%{y} pixels<extra></extra>',
    };

    const layout = {
      plot_bgcolor: '#faf9fe',
      paper_bgcolor: '#faf9fe',
      font: { family: 'Inter', color: '#1a1b1f', size: 11 },
      margin: { t: 20, r: 20, b: 45, l: 55 },
      xaxis: {
        title: 'Backscatter (dB)',
        titlefont: { size: 11 },
      },
      yaxis: {
        title: 'Pixel Count',
        titlefont: { size: 11 },
      },
      bargap: 0.05,
      showlegend: false,
      shapes: [
        // Threshold line (red dashed)
        {
          type: 'line',
          x0: threshold, x1: threshold,
          y0: 0, y1: 1,
          yref: 'paper',
          line: { color: '#dc2626', width: 2, dash: 'dash' },
        },
        // Otsu line (green dotted)
        {
          type: 'line',
          x0: otsu, x1: otsu,
          y0: 0, y1: 1,
          yref: 'paper',
          line: { color: '#16a34a', width: 2, dash: 'dot' },
        },
      ],
      annotations: [
        {
          x: threshold,
          y: 1,
          yref: 'paper',
          text: 'Threshold',
          showarrow: false,
          font: { color: '#dc2626', size: 10 },
          xanchor: threshold > otsu ? 'right' : 'left',
          yanchor: 'bottom',
          xshift: threshold > otsu ? -4 : 4,
        },
        {
          x: otsu,
          y: 0.92,
          yref: 'paper',
          text: 'Otsu',
          showarrow: false,
          font: { color: '#16a34a', size: 10 },
          xanchor: otsu > threshold ? 'right' : 'left',
          yanchor: 'bottom',
          xshift: otsu > threshold ? -4 : 4,
        },
      ],
    };

    Plotly.newPlot('sar-histogram', [barTrace], layout, {
      responsive: true,
      displayModeBar: false,
    });
  }
}
