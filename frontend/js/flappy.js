/* Flappy Bird Competition — RF Classifier Deploy */

function init_flappy() {
  // ── State ───────────────────────────────────────────────────────
  var stages = {};
  var unlockedStages = [];
  var modelStatus = null;
  var selectedMode = 'practice';
  var activeLeaderboardStage = 1;
  var leaderboardInterval = null;

  // Replay state
  var replayData = null;
  var episodeIdx = 0;
  var frameIdx = 0;
  var playing = false;
  var frameTick = 0;
  var FRAME_INTERVAL = 2;
  var animFrameId = null;
  var holdCount = 0;

  // Canvas constants
  var WORLD_W = 420;
  var WORLD_H = 580;
  var GROUND_H = 60;
  var PIPE_WIDTH = 52;
  var CANVAS_W = 800;
  var CANVAS_H = 480;
  var SCALE_X = CANVAS_W / WORLD_W;
  var SCALE_Y = CANVAS_H / WORLD_H;

  var TEAM_COLORS = [
    '#2563eb','#ef4444','#16a34a','#d97706','#8b5cf6',
    '#06b6d4','#f43f5e','#fb923c','#a3e635','#c084fc'
  ];

  // ── DOM refs ────────────────────────────────────────────────────
  var canvas = document.getElementById('replay-canvas');
  var ctx = canvas ? canvas.getContext('2d') : null;
  var canvasContainer = document.getElementById('replay-container');

  var modelStatusEl = document.getElementById('flappy-model-status');
  var stageBadgesRow = document.getElementById('flappy-stage-badges');
  var stageSelect = document.getElementById('flappy-stage-select');
  var modePracticeBtn = document.getElementById('flappy-mode-practice');
  var modeLeaderboardBtn = document.getElementById('flappy-mode-leaderboard');
  var deployBtn = document.getElementById('flappy-deploy-btn');
  var deployStatus = document.getElementById('flappy-deploy-status');
  var scoreSummary = document.getElementById('flappy-score-summary');

  var adminPassword = document.getElementById('flappy-admin-pw');
  var adminStageSelect = document.getElementById('flappy-admin-stage');
  var raceBtn = document.getElementById('flappy-race-btn');
  var raceStatus = document.getElementById('flappy-race-status');

  var btnPlay = document.getElementById('flappy-btn-play');
  var btnNext = document.getElementById('flappy-btn-next');
  var overlayStage = document.getElementById('flappy-info-stage');
  var overlayEpisode = document.getElementById('flappy-info-episode');
  var overlayScore = document.getElementById('flappy-info-score');

  var lbTabsContainer = document.getElementById('flappy-lb-tabs');
  var lbBody = document.getElementById('flappy-lb-body');

  if (!canvas || !ctx) return;

  // ── Helpers ─────────────────────────────────────────────────────

  function teamId() {
    return localStorage.getItem('earthai_team_id') || '';
  }

  function teamName() {
    return localStorage.getItem('earthai_name') || 'Student';
  }

  // ── 1. Check model status ──────────────────────────────────────

  function checkModelStatus() {
    var tid = teamId();
    if (!tid) {
      renderNoModel('No team ID found. Please log in first.');
      return;
    }

    API.get('/flappy/model-status/' + encodeURIComponent(tid)).then(function(data) {
      modelStatus = data;
      if (data && data.has_model) {
        renderModelInfo(data);
      } else {
        renderNoModel('No classifier submitted yet');
      }
    }).catch(function(err) {
      console.error('Failed to check model status:', err);
      renderNoModel('Could not check classifier status');
    });
  }

  function renderModelInfo(data) {
    if (!modelStatusEl) return;
    var f1Pct = Math.round((data.f1 || 0) * 100);
    var accPct = Math.round((data.accuracy || 0) * 100);
    var features = data.features || [];
    var featureStr = features.length > 5
      ? features.slice(0, 5).join(', ') + ' + ' + (features.length - 5) + ' more'
      : features.join(', ');

    var hp = data.hyperparameters || {};
    var precPct = Math.round((data.precision || 0) * 100);
    var recPct = Math.round((data.recall || 0) * 100);

    modelStatusEl.className = 'bg-surface-container-lowest rounded-[2rem] p-8';
    modelStatusEl.innerHTML =
      '<div class="flex items-center gap-4 mb-6">' +
        '<div class="w-12 h-12 rounded-2xl bg-emerald-50 flex items-center justify-center text-emerald-600">' +
          '<span class="material-symbols-outlined text-2xl">check_circle</span>' +
        '</div>' +
        '<div>' +
          '<p class="text-lg font-bold text-on-background negative-tracking">Your Submitted Classifier</p>' +
          '<p class="text-sm text-on-surface-variant mt-0.5">Ready to deploy as Flappy Bird agent</p>' +
        '</div>' +
      '</div>' +
      '<div class="grid grid-cols-4 gap-4 mb-6">' +
        '<div class="bg-surface-container-low rounded-2xl p-4 text-center">' +
          '<p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-1">F1 Score</p>' +
          '<p class="text-2xl font-black text-emerald-600">' + f1Pct + '%</p>' +
        '</div>' +
        '<div class="bg-surface-container-low rounded-2xl p-4 text-center">' +
          '<p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-1">Accuracy</p>' +
          '<p class="text-2xl font-black text-on-surface">' + accPct + '%</p>' +
        '</div>' +
        '<div class="bg-surface-container-low rounded-2xl p-4 text-center">' +
          '<p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-1">Precision</p>' +
          '<p class="text-2xl font-black text-on-surface">' + precPct + '%</p>' +
        '</div>' +
        '<div class="bg-surface-container-low rounded-2xl p-4 text-center">' +
          '<p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-1">Recall</p>' +
          '<p class="text-2xl font-black text-on-surface">' + recPct + '%</p>' +
        '</div>' +
      '</div>' +
      '<div class="grid grid-cols-2 gap-4">' +
        '<div class="bg-surface-container-low rounded-2xl p-4">' +
          '<p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-2">Features</p>' +
          '<p class="text-sm text-on-surface font-medium">' + featureStr + '</p>' +
        '</div>' +
        '<div class="bg-surface-container-low rounded-2xl p-4">' +
          '<p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-2">Model Config</p>' +
          '<p class="text-xs text-on-surface-variant leading-relaxed">' +
            (hp.n_trees ? 'Trees: <b>' + hp.n_trees + '</b>' : '') +
            (hp.max_depth ? ' &middot; Depth: <b>' + hp.max_depth + '</b>' : '') +
            (hp.scaling && hp.scaling !== 'none' ? ' &middot; Scaling: <b>' + hp.scaling + '</b>' : '') +
            (hp.balance && hp.balance !== 'none' ? ' &middot; Balance: <b>' + hp.balance + '</b>' : '') +
          '</p>' +
        '</div>' +
      '</div>';
  }

  function renderNoModel(msg) {
    if (!modelStatusEl) return;
    modelStatusEl.className = 'bg-error-container rounded-[2rem] p-8';
    modelStatusEl.innerHTML =
      '<div class="flex items-center justify-between">' +
        '<div class="flex items-center gap-4">' +
          '<div class="w-12 h-12 rounded-2xl bg-error/10 flex items-center justify-center text-error">' +
            '<span class="material-symbols-outlined text-2xl">warning</span>' +
          '</div>' +
          '<div>' +
            '<p class="text-lg font-bold text-on-error-container negative-tracking">' + msg + '</p>' +
            '<p class="text-sm text-on-error-container/70 mt-0.5">Train and submit a Random Forest on the Classifier page first.</p>' +
          '</div>' +
        '</div>' +
        '<a href="#classifier" class="px-6 py-2.5 bg-primary text-white rounded-full font-bold text-sm hover:bg-primary-container transition-all flex items-center gap-2 whitespace-nowrap">' +
          '<span class="material-symbols-outlined text-[18px]">arrow_forward</span>' +
          'Go to Classifier' +
        '</a>' +
      '</div>';
  }

  // ── 2. Load stages and unlocked ─────────────────────────────────

  function loadStages() {
    API.get('/flappy/stages').then(function(data) {
      stages = data.stages || {};
      renderStageSelectors();
      loadUnlockedStages();
    }).catch(function(err) {
      console.error('Failed to load flappy stages:', err);
    });
  }

  function loadUnlockedStages() {
    var tid = teamId();
    if (!tid) {
      unlockedStages = [1];
      renderStageBadges();
      renderStageSelectors();
      renderLeaderboardTabs();
      return;
    }

    API.get('/flappy/unlocked/' + encodeURIComponent(tid)).then(function(data) {
      unlockedStages = data || [1];
      renderStageBadges();
      renderStageSelectors();
      renderLeaderboardTabs();
      fetchLeaderboard();
    }).catch(function(err) {
      console.error('Failed to load unlocked stages:', err);
      unlockedStages = [1];
      renderStageBadges();
      renderStageSelectors();
      renderLeaderboardTabs();
    });
  }

  // ── 3. Stage badges ─────────────────────────────────────────────

  function renderStageBadges() {
    if (!stageBadgesRow) return;
    var stageIds = Object.keys(stages).map(Number).sort(function(a, b) { return a - b; });
    stageBadgesRow.innerHTML = stageIds.map(function(sid) {
      var s = stages[sid];
      var isUnlocked = unlockedStages.indexOf(sid) !== -1;
      var isPassed = unlockedStages.indexOf(sid + 1) !== -1;

      var bgClass, textClass, icon;
      if (isPassed) {
        bgClass = 'bg-emerald-50 border-emerald-200';
        textClass = 'text-emerald-700';
        icon = 'check_circle';
      } else if (isUnlocked) {
        bgClass = 'bg-blue-50 border-primary/20';
        textClass = 'text-primary';
        icon = 'lock_open';
      } else {
        bgClass = 'bg-slate-50 border-slate-200';
        textClass = 'text-slate-400';
        icon = 'lock';
      }

      return '<div class="flex items-center gap-2 px-4 py-2 rounded-full border ' + bgClass + ' ' + textClass + '">' +
        '<span class="material-symbols-outlined text-[16px]">' + icon + '</span>' +
        '<span class="text-xs font-bold uppercase tracking-wider">S' + sid + '</span>' +
        '<span class="text-xs font-medium">' + (s.label || '') + '</span>' +
      '</div>';
    }).join('');
  }

  // ── 4. Stage selectors ──────────────────────────────────────────

  function renderStageSelectors() {
    var stageIds = Object.keys(stages).map(Number).sort(function(a, b) { return a - b; });

    // Deploy stage select — only unlocked
    if (stageSelect) {
      stageSelect.innerHTML = stageIds.filter(function(sid) {
        return unlockedStages.indexOf(sid) !== -1;
      }).map(function(sid) {
        return '<option value="' + sid + '">Stage ' + sid + ' — ' + (stages[sid].label || '') + '</option>';
      }).join('');
    }

    // Admin stage select — all stages
    if (adminStageSelect) {
      adminStageSelect.innerHTML = stageIds.map(function(sid) {
        return '<option value="' + sid + '">Stage ' + sid + ' — ' + (stages[sid].label || '') + '</option>';
      }).join('');
    }
  }

  // ── 5. Mode toggle ──────────────────────────────────────────────

  function setMode(mode) {
    selectedMode = mode;
    if (modePracticeBtn && modeLeaderboardBtn) {
      if (mode === 'practice') {
        modePracticeBtn.className = 'flex-1 px-4 py-2.5 rounded-xl text-xs font-bold uppercase tracking-wider transition-colors bg-primary text-white';
        modeLeaderboardBtn.className = 'flex-1 px-4 py-2.5 rounded-xl text-xs font-bold uppercase tracking-wider transition-colors bg-surface-container-low text-on-surface-variant hover:bg-surface-container-high';
      } else {
        modePracticeBtn.className = 'flex-1 px-4 py-2.5 rounded-xl text-xs font-bold uppercase tracking-wider transition-colors bg-surface-container-low text-on-surface-variant hover:bg-surface-container-high';
        modeLeaderboardBtn.className = 'flex-1 px-4 py-2.5 rounded-xl text-xs font-bold uppercase tracking-wider transition-colors bg-primary text-white';
      }
    }
  }

  if (modePracticeBtn) {
    modePracticeBtn.addEventListener('click', function() { setMode('practice'); });
  }
  if (modeLeaderboardBtn) {
    modeLeaderboardBtn.addEventListener('click', function() { setMode('leaderboard'); });
  }

  // ── 6. Deploy ───────────────────────────────────────────────────

  if (deployBtn) {
    deployBtn.addEventListener('click', function() {
      var tid = teamId();
      if (!tid) {
        showDeployStatus('No team ID. Please log in first.', 'error');
        return;
      }
      if (!modelStatus || !modelStatus.has_model) {
        showDeployStatus('No classifier submitted. Go to the Classifier page first.', 'error');
        return;
      }
      if (!stageSelect) return;
      var stageId = parseInt(stageSelect.value);

      deployBtn.disabled = true;
      deployBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">hourglass_top</span> Deploying...';

      API.post('/flappy/play', {
        team_id: tid,
        stage_id: stageId,
        mode: selectedMode,
      }).then(function(data) {
        deployBtn.disabled = false;
        deployBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">smart_toy</span> Deploy My Classifier';
        showDeployStatus('', '');

        // Load replay
        loadReplay(data);

        // Show score summary
        renderScoreSummary(data);

        // Refresh unlocked
        loadUnlockedStages();
      }).catch(function(err) {
        deployBtn.disabled = false;
        deployBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">smart_toy</span> Deploy My Classifier';
        showDeployStatus('Deploy failed: ' + err.message, 'error');
      });
    });
  }

  function showDeployStatus(msg, type) {
    if (!deployStatus) return;
    if (!msg) { deployStatus.innerHTML = ''; return; }
    var color = type === 'success' ? 'text-emerald-600' : 'text-error';
    var icon = type === 'success' ? 'check_circle' : 'error';
    deployStatus.innerHTML = '<div class="flex items-center gap-2 mt-2 ' + color + '">' +
      '<span class="material-symbols-outlined text-[16px]">' + icon + '</span>' +
      '<span class="text-xs font-medium">' + msg + '</span></div>';
  }

  // ── 7. Score summary + Save to Leaderboard ──────────────────────

  var lastPlayResult = null;

  function renderScoreSummary(data) {
    if (!scoreSummary) return;
    lastPlayResult = data;
    var s = data.summary || {};
    var avgScore = s.avg_score != null ? s.avg_score : '--';
    var maxScore = s.max_score != null ? s.max_score : '--';
    var passed = s.passed;

    var passedBadge = passed
      ? '<span class="inline-flex items-center gap-1 text-xs font-bold text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full"><span class="material-symbols-outlined text-[14px]">check_circle</span>Passed</span>'
      : '<span class="inline-flex items-center gap-1 text-xs font-bold text-red-600 bg-red-50 px-3 py-1 rounded-full"><span class="material-symbols-outlined text-[14px]">cancel</span>Not Passed</span>';

    var saveBtn = '';
    if (selectedMode === 'leaderboard') {
      saveBtn = '<button id="flappy-save-lb-btn" class="w-full mt-4 bg-primary text-white rounded-full font-bold text-sm shadow-lg shadow-primary/20 py-2.5 flex items-center justify-center gap-2 hover:bg-primary-container transition-colors">' +
        '<span class="material-symbols-outlined text-[18px]">leaderboard</span> Save to Leaderboard</button>' +
        '<div id="flappy-save-lb-status"></div>';
    }

    scoreSummary.className = '';
    scoreSummary.innerHTML =
      '<div class="bg-surface-container-low rounded-2xl p-5">' +
        '<p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-3">Run Results</p>' +
        '<div class="grid grid-cols-2 gap-3 mb-3">' +
          '<div class="text-center p-3 bg-white rounded-xl border border-surface-container-high">' +
            '<p class="text-2xl font-black text-on-background tracking-tighter">' + avgScore + '</p>' +
            '<p class="text-[10px] text-on-surface-variant font-bold uppercase">Avg Score</p>' +
          '</div>' +
          '<div class="text-center p-3 bg-white rounded-xl border border-surface-container-high">' +
            '<p class="text-2xl font-black text-on-background tracking-tighter">' + maxScore + '</p>' +
            '<p class="text-[10px] text-on-surface-variant font-bold uppercase">Max Score</p>' +
          '</div>' +
        '</div>' +
        '<div class="flex justify-center">' + passedBadge + '</div>' +
        saveBtn +
      '</div>';

    // Wire save button
    var saveLbBtn = document.getElementById('flappy-save-lb-btn');
    if (saveLbBtn) {
      saveLbBtn.addEventListener('click', function() {
        saveToLeaderboard();
      });
    }
  }

  function saveToLeaderboard() {
    if (!lastPlayResult || !lastPlayResult.summary) return;
    var s = lastPlayResult.summary;
    var saveLbBtn = document.getElementById('flappy-save-lb-btn');
    var saveLbStatus = document.getElementById('flappy-save-lb-status');

    if (saveLbBtn) {
      saveLbBtn.disabled = true;
      saveLbBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">hourglass_top</span> Saving...';
    }

    API.post('/flappy/save-result', {
      team_id: teamId(),
      team_name: teamName(),
      stage_id: lastPlayResult.stage_id,
      avg_score: s.avg_score,
      max_score: s.max_score,
      episode_scores: s.scores || [],
      passed: s.passed,
    }).then(function() {
      if (saveLbBtn) {
        saveLbBtn.disabled = true;
        saveLbBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">check_circle</span> Saved';
        saveLbBtn.className = saveLbBtn.className.replace('bg-primary', 'bg-emerald-600').replace('shadow-primary/20', 'shadow-emerald-600/20');
      }
      if (saveLbStatus) {
        saveLbStatus.innerHTML = '<p class="text-xs text-emerald-600 font-medium mt-2 text-center">Result saved to leaderboard!</p>';
      }
      fetchLeaderboard();
    }).catch(function(err) {
      if (saveLbBtn) {
        saveLbBtn.disabled = false;
        saveLbBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">leaderboard</span> Save to Leaderboard';
      }
      if (saveLbStatus) {
        saveLbStatus.innerHTML = '<p class="text-xs text-error font-medium mt-2 text-center">Save failed: ' + err.message + '</p>';
      }
    });
  }

  // ── 8. Admin Race ───────────────────────────────────────────────

  if (raceBtn) {
    raceBtn.addEventListener('click', function() {
      if (!adminPassword || !adminStageSelect) return;
      var pw = adminPassword.value.trim();
      var stageId = parseInt(adminStageSelect.value);

      if (!pw) {
        showRaceStatus('Enter admin password.', 'error');
        return;
      }

      raceBtn.disabled = true;
      raceBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">hourglass_top</span> Racing...';

      API.post('/flappy/race', {
        stage_id: stageId,
        admin_password: pw,
      }).then(function(data) {
        raceBtn.disabled = false;
        raceBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">flag</span> Start Race';
        showRaceStatus('Race complete! Replay loaded.', 'success');

        // Load replay
        if (data.replay) {
          loadReplay(data.replay);
        }

        // Refresh leaderboard + unlocked stages
        loadUnlockedStages();
        fetchLeaderboard();
      }).catch(function(err) {
        raceBtn.disabled = false;
        raceBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">flag</span> Start Race';
        showRaceStatus('Race failed: ' + err.message, 'error');
      });
    });
  }

  function showRaceStatus(msg, type) {
    if (!raceStatus) return;
    var color = type === 'success' ? 'text-emerald-400' : 'text-red-400';
    var icon = type === 'success' ? 'check_circle' : 'error';
    raceStatus.innerHTML = '<div class="flex items-center gap-2 mt-3 ' + color + '">' +
      '<span class="material-symbols-outlined text-[16px]">' + icon + '</span>' +
      '<span class="text-xs font-medium">' + msg + '</span></div>';
  }

  // ── 9. Canvas Replay Viewer ─────────────────────────────────────

  function resizeCanvas() {
    if (!canvasContainer || !canvas) return;
    var w = canvasContainer.clientWidth;
    var h = canvasContainer.clientHeight || Math.round(w * 3 / 5);
    canvas.width = w;
    canvas.height = h;
    CANVAS_W = w;
    CANVAS_H = h;
    SCALE_X = CANVAS_W / WORLD_W;
    SCALE_Y = CANVAS_H / WORLD_H;
    renderFrame();
  }

  window.addEventListener('resize', resizeCanvas);

  function loadReplay(replay) {
    replayData = replay;
    episodeIdx = 0;
    frameIdx = 0;
    playing = true;
    holdCount = 0;

    // Read world dimensions from replay if available
    WORLD_W = replay.world_width || 420;
    WORLD_H = replay.world_height || 580;
    GROUND_H = replay.ground_height || 60;

    resizeCanvas();
    if (btnPlay) btnPlay.textContent = 'Pause';
    renderFrame();

    // Cancel any prior animation
    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = requestAnimationFrame(tick);
  }

  function currentEpisode() {
    if (!replayData || !replayData.episodes) return null;
    return replayData.episodes[episodeIdx] || null;
  }

  function drawBackground() {
    var grd = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
    grd.addColorStop(0, '#0e1117');
    grd.addColorStop(1, '#1a1f2e');
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
  }

  function drawGround() {
    var gy = CANVAS_H - (GROUND_H * SCALE_Y);
    ctx.fillStyle = '#1e2533';
    ctx.fillRect(0, gy, CANVAS_W, CANVAS_H - gy);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(CANVAS_W, gy);
    ctx.stroke();
  }

  // Bird is drawn at a fixed screen X (25% from left). Camera follows bird_x.
  var BIRD_SCREEN_X_RATIO = 0.25;

  function drawPipeAtScreenX(screenX, gapY, gapSize) {
    var sw = PIPE_WIDTH * SCALE_X;
    var sgapY = gapY * SCALE_Y;
    var sgapSize = gapSize * SCALE_Y;
    var capH = 8;

    var topBottom = sgapY - sgapSize / 2;
    var botTop = sgapY + sgapSize / 2;

    // Top pipe
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(screenX, 0, sw, topBottom);
    ctx.fillStyle = '#15803d';
    ctx.fillRect(screenX - 3, topBottom - capH, sw + 6, capH);

    // Bottom pipe
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(screenX, botTop, sw, CANVAS_H - botTop);
    ctx.fillStyle = '#15803d';
    ctx.fillRect(screenX - 3, botTop, sw + 6, capH);
  }

  function drawBirdAtScreenPos(screenX, screenY, alive, score) {
    var radius = 12;
    var color = TEAM_COLORS[0];

    ctx.save();
    ctx.globalAlpha = alive ? 1.0 : 0.4;

    // Body
    ctx.beginPath();
    ctx.arc(screenX, screenY, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Eye
    ctx.beginPath();
    ctx.arc(screenX + 4, screenY - 3, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(screenX + 5, screenY - 3, 2, 0, Math.PI * 2);
    ctx.fillStyle = '#0f172a';
    ctx.fill();

    // Dead marker
    if (!alive) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      var xOff = 10;
      ctx.beginPath();
      ctx.moveTo(screenX - xOff, screenY - xOff);
      ctx.lineTo(screenX + xOff, screenY + xOff);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(screenX + xOff, screenY - xOff);
      ctx.lineTo(screenX - xOff, screenY + xOff);
      ctx.stroke();
    }

    // Team name above
    ctx.font = '600 10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = color;
    ctx.fillText(teamName(), screenX, screenY - radius - 14);

    // Score below
    ctx.font = '500 10px Inter, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.fillText('Score: ' + score, screenX, screenY + radius + 14);

    ctx.restore();
  }

  function renderFrame() {
    if (!replayData || !ctx) return;

    var ep = currentEpisode();
    if (!ep || !ep.frames || ep.frames.length === 0) return;

    var frame = ep.frames[frameIdx];
    if (!frame) return;

    var birdWorldX = frame.bird_x || 0;
    var birdWorldY = frame.bird_y || 0;

    // Camera: bird stays at fixed screen X, world scrolls
    var birdScreenX = CANVAS_W * BIRD_SCREEN_X_RATIO;
    var cameraX = birdWorldX - (birdScreenX / SCALE_X);

    drawBackground();

    // Draw ALL pipes from pipe_map (visible ones only)
    var pipeMap = ep.pipe_map || [];
    var viewLeft = cameraX - 50;
    var viewRight = cameraX + WORLD_W + 50;

    for (var i = 0; i < pipeMap.length; i++) {
      var pipe = pipeMap[i];
      if (pipe.world_x >= viewLeft && pipe.world_x <= viewRight) {
        var pipeScreenX = (pipe.world_x - cameraX) * SCALE_X;
        drawPipeAtScreenX(pipeScreenX, pipe.gap_y, pipe.gap_size);
      }
    }

    drawGround();

    // Bird at fixed screen X
    var birdScreenY = birdWorldY * SCALE_Y;
    drawBirdAtScreenPos(birdScreenX, birdScreenY, frame.alive, frame.score);

    updateOverlay(frame, ep);
  }

  function updateOverlay(frame, ep) {
    var stageId = replayData.stage_id != null ? replayData.stage_id : '--';
    var totalEpisodes = replayData.episodes ? replayData.episodes.length : 0;

    if (overlayStage) overlayStage.textContent = stageId;
    if (overlayEpisode) overlayEpisode.textContent = (episodeIdx + 1) + ' / ' + totalEpisodes;
    if (overlayScore) overlayScore.textContent = frame.score;
  }

  function tick() {
    animFrameId = requestAnimationFrame(tick);

    if (!playing || !replayData) return;

    frameTick++;
    if (frameTick < FRAME_INTERVAL) return;
    frameTick = 0;

    var ep = currentEpisode();
    if (!ep || !ep.frames) return;

    if (frameIdx < ep.frames.length - 1) {
      frameIdx++;
    } else {
      holdCount++;
      if (holdCount >= 30) {
        holdCount = 0;
        advanceEpisode();
      }
    }

    renderFrame();
  }

  function advanceEpisode() {
    if (!replayData) return;
    if (episodeIdx < replayData.episodes.length - 1) {
      episodeIdx++;
    } else {
      episodeIdx = 0;
    }
    frameIdx = 0;
  }

  // Play/Pause
  if (btnPlay) {
    btnPlay.addEventListener('click', function() {
      playing = !playing;
      btnPlay.textContent = playing ? 'Pause' : 'Play';
      if (playing) renderFrame();
    });
  }

  // Next Episode
  if (btnNext) {
    btnNext.addEventListener('click', function() {
      holdCount = 0;
      advanceEpisode();
      frameIdx = 0;
      renderFrame();
    });
  }

  // Draw idle canvas
  function drawIdleCanvas() {
    if (!ctx) return;
    resizeCanvas();
    var grd = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
    grd.addColorStop(0, '#0e1117');
    grd.addColorStop(1, '#1a1f2e');
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    ctx.fillStyle = '#64748b';
    ctx.font = '500 14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Deploy your classifier to see replay here', CANVAS_W / 2, CANVAS_H / 2 - 10);
    ctx.font = '400 12px Inter, sans-serif';
    ctx.fillStyle = '#475569';
    ctx.fillText('Select a stage, choose a mode, and click Deploy', CANVAS_W / 2, CANVAS_H / 2 + 14);
  }

  // ── 10. Leaderboard ─────────────────────────────────────────────

  function renderLeaderboardTabs() {
    if (!lbTabsContainer) return;
    var stageIds = Object.keys(stages).map(Number).sort(function(a, b) { return a - b; });
    lbTabsContainer.innerHTML = stageIds.map(function(sid) {
      var active = sid === activeLeaderboardStage;
      var cls = active
        ? 'bg-primary text-white'
        : 'bg-surface-container-low text-on-surface-variant hover:bg-surface-container-high';
      return '<button data-lb-stage="' + sid + '" class="px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-colors ' + cls + '">S' + sid + '</button>';
    }).join('');

    // Wire clicks
    var btns = lbTabsContainer.querySelectorAll('button[data-lb-stage]');
    for (var i = 0; i < btns.length; i++) {
      btns[i].addEventListener('click', function() {
        activeLeaderboardStage = parseInt(this.getAttribute('data-lb-stage'));
        renderLeaderboardTabs();
        fetchLeaderboard();
      });
    }
  }

  function fetchLeaderboard() {
    API.get('/leaderboard/flappy/' + activeLeaderboardStage).then(function(entries) {
      renderLeaderboard(entries);
    }).catch(function(err) {
      console.error('Flappy leaderboard fetch error:', err);
    });
  }

  function renderLeaderboard(entries) {
    if (!lbBody) return;
    if (!entries || entries.length === 0) {
      lbBody.innerHTML = '<div class="text-center py-8 text-on-surface-variant text-sm">No submissions for this stage yet</div>';
      return;
    }

    var myTeam = teamName();
    lbBody.innerHTML = entries.map(function(e, i) {
      var rank = i + 1;
      var isMe = e.team_name === myTeam;
      var bgClass = isMe ? 'bg-blue-50 border-primary/20' : 'bg-white border-slate-100';
      var initial = (e.team_name || '?').charAt(0).toUpperCase();
      var color = TEAM_COLORS[i % TEAM_COLORS.length];
      var passedBadge = e.passed
        ? '<span class="text-[9px] font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded-full uppercase">Passed</span>'
        : '';

      return '<div class="flex items-center gap-3 p-3 rounded-2xl border ' + bgClass + '">' +
        '<span class="text-lg font-black w-6 italic ' + (rank <= 3 ? 'text-primary/40' : 'text-slate-300') + '">' + (rank < 10 ? '0' : '') + rank + '</span>' +
        '<div class="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-xs flex-shrink-0" style="background:' + color + '">' + initial + '</div>' +
        '<div class="flex-1 min-w-0">' +
          '<p class="text-sm font-bold text-on-surface truncate">' + e.team_name + (isMe ? ' <span class="text-[10px] text-primary font-bold">(you)</span>' : '') + '</p>' +
          '<p class="text-[10px] text-on-surface-variant">Avg: ' + (e.avg_score != null ? e.avg_score.toFixed(1) : '--') + ' | Max: ' + (e.max_score != null ? e.max_score.toFixed(1) : '--') + '</p>' +
        '</div>' +
        '<div class="text-right flex flex-col items-end gap-1">' +
          '<p class="font-black text-on-surface tracking-tighter">' + (e.avg_score != null ? e.avg_score.toFixed(1) : '--') + '</p>' +
          passedBadge +
        '</div>' +
      '</div>';
    }).join('');
  }

  // Poll leaderboard every 5 seconds
  leaderboardInterval = setInterval(fetchLeaderboard, 5000);

  // Clean up interval when page navigates away
  var observer = new MutationObserver(function() {
    if (!document.getElementById('replay-canvas')) {
      clearInterval(leaderboardInterval);
      if (animFrameId) cancelAnimationFrame(animFrameId);
      observer.disconnect();
    }
  });
  var pageContent = document.getElementById('page-content');
  if (pageContent) {
    observer.observe(pageContent, { childList: true });
  }

  // ── Init ────────────────────────────────────────────────────────
  drawIdleCanvas();
  checkModelStatus();
  loadStages();
}
