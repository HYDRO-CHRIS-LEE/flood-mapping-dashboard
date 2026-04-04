/* AI Flood Classifier page logic */

function init_classifier() {
  const featureList = document.getElementById('clf-feature-list');
  const trainBtn    = document.getElementById('clf-train-btn');
  const resultsDiv  = document.getElementById('clf-results');

  // Slider elements
  const sampleSlider = document.getElementById('clf-sample-pct');
  const sampleVal    = document.getElementById('clf-sample-val');
  const treesSlider  = document.getElementById('clf-n-trees');
  const treesVal     = document.getElementById('clf-trees-val');
  const depthSlider  = document.getElementById('clf-max-depth');
  const depthVal     = document.getElementById('clf-depth-val');
  const leafSlider   = document.getElementById('clf-min-leaf');
  const leafVal      = document.getElementById('clf-leaf-val');

  if (!featureList || !trainBtn || !resultsDiv) return;

  // ── State ───────────────────────────────────────────────────
  let featureInfo = [];
  let lastMetrics = null;
  let lastImportance = null;
  let lastFeatures = [];
  let leaderboardInterval = null;

  // Default checked features
  const DEFAULT_CHECKED = ['NDWI', 'elevation', 'slope'];

  // ── Wire slider labels ──────────────────────────────────────
  sampleSlider.addEventListener('input', () => {
    sampleVal.textContent = sampleSlider.value + '%';
  });
  treesSlider.addEventListener('input', () => {
    treesVal.textContent = treesSlider.value;
  });
  depthSlider.addEventListener('input', () => {
    depthVal.textContent = depthSlider.value;
  });
  leafSlider.addEventListener('input', () => {
    leafVal.textContent = leafSlider.value;
  });

  // ── Fetch feature info ──────────────────────────────────────
  API.get('/classifier/features').then(data => {
    featureInfo = data.features;
    renderFeatureCheckboxes();
  }).catch(err => {
    featureList.innerHTML = `<p class="text-error text-sm">Failed to load features: ${err.message}</p>`;
  });

  function renderFeatureCheckboxes() {
    featureList.innerHTML = featureInfo.map(f => {
      const checked = DEFAULT_CHECKED.includes(f.key) ? 'checked' : '';
      return `
        <label class="flex items-center gap-3 rounded-xl p-3 bg-surface-container-low cursor-pointer hover:bg-surface-container-high transition-colors">
          <input type="checkbox" name="clf-feature" value="${f.key}" ${checked} class="accent-primary rounded" />
          <div class="w-8 h-8 rounded-lg flex items-center justify-center bg-blue-50 text-primary flex-shrink-0">
            <span class="material-symbols-outlined text-[18px]">${f.icon}</span>
          </div>
          <div class="min-w-0">
            <p class="text-sm font-semibold text-on-surface">${f.short}</p>
            <p class="text-xs text-on-surface-variant truncate">${f.description}</p>
          </div>
        </label>`;
    }).join('');
  }

  // ── Train button ────────────────────────────────────────────
  trainBtn.addEventListener('click', () => {
    const selectedFeatures = Array.from(
      document.querySelectorAll('input[name="clf-feature"]:checked')
    ).map(cb => cb.value);

    if (selectedFeatures.length === 0) {
      alert('Please select at least one feature.');
      return;
    }

    const formData = {
      team_id: localStorage.getItem('earthai_team_id') || '',
      features: selectedFeatures,
      n_trees: parseInt(treesSlider.value),
      max_depth: parseInt(depthSlider.value),
      min_samples_leaf: parseInt(leafSlider.value),
      max_features: document.getElementById('clf-max-features').value,
      bootstrap: document.getElementById('clf-bootstrap').checked,
      class_weight: document.getElementById('clf-class-weight').checked,
      scaling: document.getElementById('clf-scaling').value,
      outlier: document.getElementById('clf-outlier').value,
      balance: document.getElementById('clf-balance').value,
      sample_pct: parseInt(sampleSlider.value),
    };

    // Show loading state
    trainBtn.disabled = true;
    trainBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">hourglass_top</span> Training...';

    API.post('/classifier/train', formData).then(data => {
      trainBtn.disabled = false;
      trainBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">model_training</span> Train Model';

      lastMetrics = data.metrics;
      lastImportance = data.importance;
      lastFeatures = selectedFeatures;

      renderResults(data.metrics, data.importance, data.hints);
    }).catch(err => {
      trainBtn.disabled = false;
      trainBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">model_training</span> Train Model';

      resultsDiv.innerHTML = `
        <div class="bg-error-container rounded-[2rem] p-8 flex items-center gap-4">
          <span class="material-symbols-outlined text-3xl text-error">error</span>
          <div>
            <p class="text-sm font-bold text-error">Training Failed</p>
            <p class="text-sm text-error">${err.message}</p>
          </div>
        </div>`;
    });
  });

  // ── Render results ──────────────────────────────────────────
  function renderResults(metrics, importance, hints) {
    // Color-code F1
    let f1Color = '#ba1a1a';
    if (metrics.f1 >= 0.8) f1Color = '#16a34a';
    else if (metrics.f1 >= 0.6) f1Color = '#d97706';

    resultsDiv.innerHTML = `
      <!-- Metric cards -->
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-surface-container-lowest rounded-[2rem] p-6 text-center">
          <p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-2">F1 Score</p>
          <p class="text-4xl font-black tracking-tighter" style="color:${f1Color}">${metrics.f1.toFixed(4)}</p>
        </div>
        <div class="bg-surface-container-lowest rounded-[2rem] p-6 text-center">
          <p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-2">Accuracy</p>
          <p class="text-4xl font-black tracking-tighter text-primary">${metrics.accuracy.toFixed(4)}</p>
        </div>
        <div class="bg-surface-container-lowest rounded-[2rem] p-6 text-center">
          <p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-2">Precision</p>
          <p class="text-4xl font-black tracking-tighter text-primary">${metrics.precision.toFixed(4)}</p>
        </div>
        <div class="bg-surface-container-lowest rounded-[2rem] p-6 text-center">
          <p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-2">Recall</p>
          <p class="text-4xl font-black tracking-tighter text-primary">${metrics.recall.toFixed(4)}</p>
        </div>
      </div>

      <!-- Train/Test info -->
      <div class="bg-surface-container-lowest rounded-[2rem] p-6">
        <div class="flex flex-wrap gap-6 text-sm text-on-surface-variant">
          <span><span class="font-bold text-on-surface">${metrics.n_train.toLocaleString()}</span> training samples</span>
          <span><span class="font-bold text-on-surface">${metrics.n_test.toLocaleString()}</span> test samples</span>
          <span>Test events: ${metrics.test_events.join(', ')}</span>
        </div>
      </div>

      <!-- Charts row -->
      <div class="grid grid-cols-2 gap-4">
        <!-- Confusion Matrix -->
        <div class="bg-surface-container-lowest rounded-[2rem] p-6">
          <p class="text-[10px] font-bold text-secondary uppercase tracking-widest mb-3">Confusion Matrix</p>
          <div id="clf-cm-chart" style="width:100%;height:220px;"></div>
        </div>
        <!-- Feature Importance -->
        <div class="bg-surface-container-lowest rounded-[2rem] p-6">
          <p class="text-[10px] font-bold text-secondary uppercase tracking-widest mb-3">Feature Importance</p>
          <div id="clf-importance-chart" style="width:100%;"></div>
        </div>
      </div>

      <!-- Hints -->
      <div id="clf-hints"></div>

      <!-- Submit button -->
      <div class="flex justify-end">
        <button id="clf-submit-btn" class="bg-primary text-white rounded-full font-bold text-sm shadow-lg shadow-primary/20 px-8 py-3 flex items-center gap-2 hover:bg-primary-container transition-colors">
          <span class="material-symbols-outlined text-[18px]">leaderboard</span>
          Submit to Leaderboard
        </button>
      </div>
    `;

    // Render charts
    renderConfusionMatrix(metrics.cm);
    renderFeatureImportance(importance);
    renderHints(hints);

    // Wire submit button
    const submitBtn = document.getElementById('clf-submit-btn');
    if (submitBtn) {
      submitBtn.addEventListener('click', submitToLeaderboard);
    }
  }

  // ── Confusion Matrix ────────────────────────────────────────
  function renderConfusionMatrix(cm) {
    const labels = ['Non-flood', 'Flood'];
    // cm is [[TN, FP], [FN, TP]]
    const z = cm;
    const text = cm.map(row => row.map(v => v.toString()));

    const trace = {
      z: z,
      x: labels,
      y: labels,
      type: 'heatmap',
      colorscale: [[0, '#d8e2ff'], [1, '#004493']],
      text: text,
      texttemplate: '%{text}',
      textfont: { size: 16, color: '#fff' },
      hovertemplate: 'Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
      showscale: false,
    };

    const layout = {
      paper_bgcolor: '#faf9fe',
      plot_bgcolor: '#faf9fe',
      font: { family: 'Inter', color: '#1a1b1f', size: 11 },
      height: 220,
      margin: { t: 10, r: 10, b: 40, l: 70 },
      xaxis: { title: 'Predicted', side: 'bottom' },
      yaxis: { title: 'Actual', autorange: 'reversed' },
    };

    Plotly.newPlot('clf-cm-chart', [trace], layout, {
      responsive: true,
      displayModeBar: false,
    });
  }

  // ── Feature Importance ──────────────────────────────────────
  function renderFeatureImportance(importance) {
    // Sort ascending for horizontal bar
    const entries = Object.entries(importance).sort((a, b) => a[1] - b[1]);
    const features = entries.map(e => e[0]);
    const values = entries.map(e => e[1]);

    // Blue gradient based on value
    const maxVal = Math.max(...values, 0.001);
    const colors = values.map(v => {
      const ratio = v / maxVal;
      const r = Math.round(0 + (37 * ratio));
      const g = Math.round(70 + (99 * ratio));
      const b = Math.round(180 + (55 * ratio));
      return `rgb(${r},${g},${b})`;
    });

    const trace = {
      y: features,
      x: values,
      type: 'bar',
      orientation: 'h',
      marker: { color: colors },
      hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
    };

    const chartHeight = Math.max(180, features.length * 38);
    const container = document.getElementById('clf-importance-chart');
    if (container) container.style.height = chartHeight + 'px';

    const layout = {
      paper_bgcolor: '#faf9fe',
      plot_bgcolor: '#faf9fe',
      font: { family: 'Inter', color: '#1a1b1f', size: 11 },
      height: chartHeight,
      margin: { t: 10, r: 20, b: 30, l: 100 },
      xaxis: { title: 'Importance' },
      yaxis: { automargin: true },
    };

    Plotly.newPlot('clf-importance-chart', [trace], layout, {
      responsive: true,
      displayModeBar: false,
    });
  }

  // ── Hints ───────────────────────────────────────────────────
  function renderHints(hints) {
    const container = document.getElementById('clf-hints');
    if (!container || !hints || hints.length === 0) return;

    container.innerHTML = hints.map(hint => `
      <div class="bg-blue-50 border border-blue-200 rounded-2xl p-5 flex items-start gap-3">
        <span class="material-symbols-outlined text-primary mt-0.5">lightbulb</span>
        <p class="text-sm text-on-surface leading-relaxed">${hint}</p>
      </div>
    `).join('');
  }

  // ── Submit to leaderboard ───────────────────────────────────
  function submitToLeaderboard() {
    if (!lastMetrics) return;

    const submitBtn = document.getElementById('clf-submit-btn');
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">hourglass_top</span> Submitting...';
    }

    const teamId = localStorage.getItem('earthai_team_id') || '';
    const teamName = localStorage.getItem('earthai_name') || 'Student';

    API.post('/classifier/submit', {
      team_id: teamId,
      team_name: teamName,
      f1: lastMetrics.f1,
      accuracy: lastMetrics.accuracy,
      precision_val: lastMetrics.precision,
      recall: lastMetrics.recall,
      features: lastFeatures,
      n_trees: parseInt(treesSlider.value),
      max_depth: parseInt(depthSlider.value),
    }).then(data => {
      if (submitBtn) {
        submitBtn.disabled = false;
        if (data.status === 'kept_existing') {
          submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">info</span> Existing Score Higher';
          setTimeout(() => {
            submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">leaderboard</span> Submit to Leaderboard';
          }, 3000);
        } else {
          submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">check_circle</span> Submitted!';
          setTimeout(() => {
            submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">leaderboard</span> Submit to Leaderboard';
          }, 3000);
        }
      }
      // Immediately refresh leaderboard
      fetchLeaderboard();
    }).catch(err => {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">error</span> Failed';
        setTimeout(() => {
          submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">leaderboard</span> Submit to Leaderboard';
        }, 3000);
      }
      console.error('Submit error:', err);
    });
  }

  // ── Leaderboard polling ─────────────────────────────────────
  function fetchLeaderboard() {
    API.get('/leaderboard/classifier').then(entries => {
      renderLeaderboard(entries);
    }).catch(err => {
      console.error('Leaderboard fetch error:', err);
    });
  }

  function renderLeaderboard(entries) {
    const tbody = document.getElementById('clf-lb-body');
    if (!tbody) return;

    if (!entries || entries.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="text-center py-8 text-on-surface-variant">No submissions yet</td></tr>';
      return;
    }

    const myTeamId = localStorage.getItem('earthai_team_id') || '';

    tbody.innerHTML = entries.map((e, i) => {
      const isMe = e.team_id === myTeamId;
      const rowClass = isMe ? 'bg-blue-50' : (i % 2 === 0 ? '' : 'bg-surface-container-low');
      const rank = i + 1;
      const medal = rank === 1 ? '\u{1F947}' : rank === 2 ? '\u{1F948}' : rank === 3 ? '\u{1F949}' : rank;
      const timeStr = e.submitted_at ? new Date(e.submitted_at).toLocaleString() : '';

      return `
        <tr class="${rowClass} border-b border-outline-variant/50">
          <td class="py-3 px-2 font-bold">${medal}</td>
          <td class="py-3 px-2 font-semibold text-on-surface">${isMe ? (localStorage.getItem('earthai_name') || e.team_name) : e.team_name}${isMe ? ' <span class="text-[10px] text-primary font-bold">(you)</span>' : ''}</td>
          <td class="py-3 px-2 font-black text-primary">${e.f1 != null ? e.f1.toFixed(4) : '--'}</td>
          <td class="py-3 px-2">${e.accuracy != null ? e.accuracy.toFixed(4) : '--'}</td>
          <td class="py-3 px-2">${e.precision_val != null ? e.precision_val.toFixed(4) : '--'}</td>
          <td class="py-3 px-2">${e.recall != null ? e.recall.toFixed(4) : '--'}</td>
          <td class="py-3 px-2">${e.n_trees || '--'}</td>
          <td class="py-3 px-2">${e.max_depth || '--'}</td>
          <td class="py-3 px-2 text-xs text-on-surface-variant">${timeStr}</td>
        </tr>`;
    }).join('');
  }

  // Start polling leaderboard
  fetchLeaderboard();
  leaderboardInterval = setInterval(fetchLeaderboard, 5000);

  // Clean up interval when page navigates away
  // (The SPA replaces innerHTML, which destroys the page, but the interval persists)
  // We use a MutationObserver on the parent to detect removal
  const observer = new MutationObserver(() => {
    if (!document.getElementById('clf-results')) {
      clearInterval(leaderboardInterval);
      observer.disconnect();
    }
  });
  const pageContent = document.getElementById('page-content');
  if (pageContent) {
    observer.observe(pageContent, { childList: true });
  }
}
