/* Rainfall Analysis page logic */

function init_rainfall() {
  const selector = document.getElementById('event-select');
  const content  = document.getElementById('rainfall-content');
  const loadBtn  = document.getElementById('rainfall-load-btn');

  if (!selector || !content) return;

  // ── Populate event dropdown ─────────────────────────────────
  API.get('/events').then(events => {
    events.forEach(ev => {
      const opt = document.createElement('option');
      opt.value = ev.key;
      opt.textContent = `${ev.label} (${ev.year})`;
      selector.appendChild(opt);
    });
    // Auto-load first event
    if (events.length > 0) {
      selector.value = events[0].key;
      loadEvent(events[0].key);
    }
  }).catch(err => {
    content.innerHTML = `<p class="text-error text-sm">Failed to load events: ${err.message}</p>`;
  });

  // ── Event handlers ──────────────────────────────────────────
  selector.addEventListener('change', () => {
    if (selector.value) loadEvent(selector.value);
  });

  if (loadBtn) {
    loadBtn.addEventListener('click', () => {
      if (selector.value) loadEvent(selector.value);
    });
  }

  // ── State ───────────────────────────────────────────────────
  let rainfallData = null;
  let threshold = 20;

  // ── Load event data ─────────────────────────────────────────
  function loadEvent(key) {
    content.innerHTML = `
      <div class="flex items-center justify-center py-16">
        <div class="text-center">
          <span class="material-symbols-outlined mb-3 text-4xl text-outline animate-pulse">hourglass_top</span>
          <p class="text-sm text-on-surface-variant">Loading rainfall data...</p>
        </div>
      </div>`;

    API.get(`/rainfall/${key}`).then(data => {
      rainfallData = data;
      renderDashboard();
    }).catch(err => {
      content.innerHTML = `
        <div class="bg-error-container rounded-[2rem] p-8 text-center">
          <span class="material-symbols-outlined text-3xl text-error mb-2">error</span>
          <p class="text-sm text-error">${err.message}</p>
        </div>`;
    });
  }

  // ── Render full dashboard ───────────────────────────────────
  function renderDashboard() {
    const d = rainfallData;
    content.innerHTML = `
      <!-- Controls -->
      <div class="bg-surface-container-lowest rounded-[2rem] p-6 mb-4">
        <div class="flex flex-wrap items-center gap-6">
          <div class="flex items-center gap-3">
            <label class="text-sm font-semibold text-on-surface-variant">Threshold</label>
            <input id="rf-threshold" type="range" min="5" max="80" value="${threshold}" class="w-40 accent-primary" />
            <span id="rf-threshold-val" class="text-sm font-bold text-primary w-14">${threshold} mm</span>
          </div>
          <label class="flex items-center gap-2 text-sm text-on-surface-variant cursor-pointer">
            <input id="rf-show-line" type="checkbox" class="accent-primary rounded" /> Threshold line
          </label>
          <label class="flex items-center gap-2 text-sm text-on-surface-variant cursor-pointer">
            <input id="rf-show-flood" type="checkbox" class="accent-primary rounded" /> Flood period
          </label>
          <label class="flex items-center gap-2 text-sm text-on-surface-variant cursor-pointer">
            <input id="rf-show-cumul" type="checkbox" class="accent-primary rounded" /> Cumulative overlay
          </label>
        </div>
      </div>

      <!-- Bento grid -->
      <div class="grid grid-cols-12 gap-4">
        <!-- Chart: col 8 -->
        <div class="col-span-12 lg:col-span-8 bg-surface-container-lowest rounded-[2rem] p-8">
          <div id="rf-chart" style="width:100%;height:400px;"></div>
        </div>

        <!-- Stats cards: col 4 -->
        <div class="col-span-12 lg:col-span-4 flex flex-col gap-4">
          <!-- Cumulative -->
          <div class="bg-primary rounded-[2rem] p-8 text-white flex-1 flex flex-col justify-center">
            <p class="text-white/70 font-bold text-sm uppercase tracking-widest mb-2">Cumulative Precipitation</p>
            <p class="text-6xl font-black negative-tracking leading-none">${Math.round(d.total_mm)}<span class="text-2xl font-bold ml-1">mm</span></p>
            <p class="text-white/70 text-sm mt-2">${d.period_days} days observed</p>
          </div>

          <!-- Peak -->
          <div class="bg-primary rounded-[2rem] p-8 text-white flex-1 flex flex-col justify-center">
            <p class="text-white/70 font-bold text-sm uppercase tracking-widest mb-2">Peak Rainfall</p>
            <p class="text-6xl font-black negative-tracking leading-none">${d.peak_val}<span class="text-2xl font-bold ml-1">mm</span></p>
            <p class="text-white/70 text-sm mt-2">${d.peak_date}</p>
          </div>
        </div>

        <!-- Fact card: col 8 on next row, or col 4 side -->
        <div class="col-span-12 lg:col-span-4 bg-surface-container-lowest rounded-[2rem] p-8">
          <p class="text-secondary font-bold text-sm uppercase tracking-widest mb-3">Event Fact</p>
          <p class="text-on-surface text-base leading-relaxed">${d.fact || 'No additional fact available for this event.'}</p>
          <p class="mt-4 text-sm text-on-surface-variant">${d.event.label} &mdash; ${d.event.region}, ${d.event.year}</p>
        </div>

        <!-- Empty col 8 for grid balance -->
        <div class="col-span-12 lg:col-span-8 bg-surface-container-lowest rounded-[2rem] p-8">
          <p class="text-secondary font-bold text-sm uppercase tracking-widest mb-3">About This Chart</p>
          <p class="text-on-surface-variant text-sm leading-relaxed">
            Daily precipitation from GPM (Global Precipitation Measurement). Red bars indicate days exceeding the threshold.
            Use the controls above to toggle the threshold line, flood period highlight, and cumulative overlay.
          </p>
        </div>
      </div>`;

    // Wire up controls
    const sliderEl  = document.getElementById('rf-threshold');
    const valEl     = document.getElementById('rf-threshold-val');
    const lineEl    = document.getElementById('rf-show-line');
    const floodEl   = document.getElementById('rf-show-flood');
    const cumulEl   = document.getElementById('rf-show-cumul');

    sliderEl.addEventListener('input', () => {
      threshold = parseInt(sliderEl.value);
      valEl.textContent = threshold + ' mm';
      updateChart();
    });
    lineEl.addEventListener('change', updateChart);
    floodEl.addEventListener('change', updateChart);
    cumulEl.addEventListener('change', updateChart);

    updateChart();
  }

  // ── Update / render Plotly chart ────────────────────────────
  function updateChart() {
    const d = rainfallData;
    if (!d) return;

    const showLine  = document.getElementById('rf-show-line')?.checked || false;
    const showFlood = document.getElementById('rf-show-flood')?.checked || false;
    const showCumul = document.getElementById('rf-show-cumul')?.checked || false;

    // Bar colors
    const colors = d.precip.map(v => v >= threshold ? '#ba1a1a' : '#adc6ff');

    const traces = [];

    // Main bar trace
    traces.push({
      x: d.dates,
      y: d.precip,
      type: 'bar',
      marker: { color: colors },
      name: 'Daily Precip (mm)',
      hovertemplate: '%{x}<br>%{y:.1f} mm<extra></extra>',
    });

    // Cumulative overlay
    if (showCumul) {
      let cumul = [];
      let running = 0;
      d.precip.forEach(v => { running += v; cumul.push(Math.round(running * 10) / 10); });
      traces.push({
        x: d.dates,
        y: cumul,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#4f46e5', width: 2, dash: 'dot' },
        name: 'Cumulative (mm)',
        yaxis: 'y2',
        hovertemplate: '%{x}<br>%{y:.0f} mm cumul.<extra></extra>',
      });
    }

    // Layout
    const shapes = [];
    const annotations = [];

    // Threshold line
    if (showLine) {
      shapes.push({
        type: 'line',
        x0: d.dates[0],
        x1: d.dates[d.dates.length - 1],
        y0: threshold,
        y1: threshold,
        line: { color: '#d97706', width: 2, dash: 'dash' },
      });
      annotations.push({
        x: d.dates[d.dates.length - 1],
        y: threshold,
        text: `${threshold} mm`,
        showarrow: false,
        font: { color: '#d97706', size: 11 },
        xanchor: 'left',
        yanchor: 'bottom',
      });
    }

    // Flood period rect
    if (showFlood && d.flood_window) {
      shapes.push({
        type: 'rect',
        x0: d.flood_window[0],
        x1: d.flood_window[1],
        y0: 0,
        y1: 1,
        yref: 'paper',
        fillcolor: 'rgba(186,26,26,0.08)',
        line: { width: 0 },
        layer: 'below',
      });
      annotations.push({
        x: d.flood_window[0],
        y: 1,
        yref: 'paper',
        text: 'Flood period',
        showarrow: false,
        font: { color: '#ba1a1a', size: 10 },
        xanchor: 'left',
        yanchor: 'top',
      });
    }

    const layout = {
      plot_bgcolor: '#faf9fe',
      paper_bgcolor: '#faf9fe',
      font: { family: 'Inter', color: '#1a1b1f' },
      margin: { t: 30, r: showCumul ? 60 : 20, b: 50, l: 50 },
      xaxis: { title: '', tickangle: -45, tickfont: { size: 10 } },
      yaxis: { title: 'Precipitation (mm)', rangemode: 'tozero' },
      shapes,
      annotations,
      showlegend: traces.length > 1,
      legend: { x: 0, y: 1.12, orientation: 'h' },
      bargap: 0.15,
    };

    if (showCumul) {
      layout.yaxis2 = {
        title: 'Cumulative (mm)',
        overlaying: 'y',
        side: 'right',
        rangemode: 'tozero',
        showgrid: false,
      };
    }

    const config = { responsive: true, displayModeBar: false };

    Plotly.newPlot('rf-chart', traces, layout, config);
  }
}
