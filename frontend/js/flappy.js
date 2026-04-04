/* Flappy Bird Competition — Student Practice + Admin Competition */

function init_flappy() {
  // ── State ───────────────────────────────────────────────────────
  var stages = {};
  var unlockedStages = [];
  var modelStatus = null;
  var activeMode = 'student'; // 'student' | 'admin'

  // Student replay state
  var replayData = null;
  var episodeIdx = 0;
  var frameIdx = 0;
  var playing = false;
  var frameTick = 0;
  var FRAME_INTERVAL = 2;
  var animFrameId = null;
  var holdCount = 0;
  var lastPlayResult = null;
  var stageResults = {};

  // Admin state
  var adminPw = '';
  var competitionResult = null;
  var adminStageIdx = 0;       // index into stage_results array
  var adminFrameIdx = 0;
  var adminPlaying = false;
  var adminFrameTick = 0;
  var adminAnimFrameId = null;
  var adminHoldCount = 0;
  var ADMIN_FRAME_INTERVAL = 2;
  var adminEliminationList = []; // {team_name, score, stage, rank}

  // Canvas constants (shared)
  var WORLD_W = 420;
  var WORLD_H = 580;
  var GROUND_H = 60;
  var PIPE_WIDTH = 52;
  var CANVAS_W = 800;
  var CANVAS_H = 480;
  var SCALE_X = CANVAS_W / WORLD_W;
  var SCALE_Y = CANVAS_H / WORLD_H;
  var BIRD_SCREEN_X_RATIO = 0.25;

  // Admin canvas dimensions (separate)
  var ADMIN_CANVAS_W = 800;
  var ADMIN_CANVAS_H = 480;
  var ADMIN_SCALE_X = ADMIN_CANVAS_W / WORLD_W;
  var ADMIN_SCALE_Y = ADMIN_CANVAS_H / WORLD_H;

  var TEAM_COLORS = [
    '#2563eb','#ef4444','#16a34a','#d97706','#8b5cf6',
    '#06b6d4','#f43f5e','#fb923c','#a3e635','#c084fc',
    '#14b8a6','#e11d48','#f59e0b','#6366f1','#84cc16',
    '#0ea5e9','#ec4899','#f97316','#a855f7','#10b981'
  ];

  // ── DOM refs — Student ──────────────────────────────────────────
  var tabStudent = document.getElementById('flappy-tab-student');
  var tabAdmin = document.getElementById('flappy-tab-admin');
  var studentView = document.getElementById('flappy-student-view');
  var adminView = document.getElementById('flappy-admin-view');

  var canvas = document.getElementById('replay-canvas');
  var ctx = canvas ? canvas.getContext('2d') : null;
  var canvasContainer = document.getElementById('replay-container');

  var modelStatusEl = document.getElementById('flappy-model-status');
  var stageSelect = document.getElementById('flappy-stage-select');
  var deployBtn = document.getElementById('flappy-deploy-btn');
  var deployStatus = document.getElementById('flappy-deploy-status');

  var submitCompBtn = document.getElementById('flappy-submit-competition-btn');
  var submitCompStatus = document.getElementById('flappy-submit-competition-status');

  var btnPlay = document.getElementById('flappy-btn-play');
  var btnNext = document.getElementById('flappy-btn-next');
  var overlayStage = document.getElementById('flappy-info-stage');
  var overlayScore = document.getElementById('flappy-info-score');

  // ── DOM refs — Admin ────────────────────────────────────────────
  var adminLoginPanel = document.getElementById('admin-login-panel');
  var adminPasswordInput = document.getElementById('admin-password');
  var adminLoginBtn = document.getElementById('admin-login-btn');
  var adminLoginError = document.getElementById('admin-login-error');

  var adminCompPanel = document.getElementById('admin-competition-panel');
  var adminTeamsList = document.getElementById('admin-teams-list');
  var adminRefreshTeams = document.getElementById('admin-refresh-teams');
  var adminStartBtn = document.getElementById('admin-start-btn');
  var adminStartStatus = document.getElementById('admin-start-status');

  var adminCanvas = document.getElementById('admin-replay-canvas');
  var adminCtx = adminCanvas ? adminCanvas.getContext('2d') : null;
  var adminCanvasContainer = document.getElementById('admin-replay-container');

  var adminBtnPlay = document.getElementById('admin-btn-play');
  var adminBtnNextStage = document.getElementById('admin-btn-next-stage');
  var adminInfoStage = document.getElementById('admin-info-stage');
  var adminInfoAlive = document.getElementById('admin-info-alive');
  var adminInfoScore = document.getElementById('admin-info-score');

  var adminEliminationEl = document.getElementById('admin-elimination-list');
  var adminFinalRanking = document.getElementById('admin-final-ranking');

  if (!canvas || !ctx) return;

  // ── Helpers ─────────────────────────────────────────────────────

  function teamId() {
    return localStorage.getItem('earthai_team_id') || '';
  }

  function teamName() {
    return localStorage.getItem('earthai_name') || 'Student';
  }

  // ── Mode Tabs ───────────────────────────────────────────────────

  function setActiveMode(mode) {
    activeMode = mode;
    var activeClass = 'px-6 py-2.5 rounded-full font-bold text-sm transition-all bg-primary text-white shadow-lg shadow-primary/20';
    var inactiveClass = 'px-6 py-2.5 rounded-full font-bold text-sm transition-all bg-surface-container-high text-on-surface hover:bg-surface-container-highest';

    if (tabStudent) tabStudent.className = mode === 'student' ? activeClass : inactiveClass;
    if (tabAdmin) tabAdmin.className = mode === 'admin' ? activeClass : inactiveClass;

    if (studentView) {
      if (mode === 'student') { studentView.classList.remove('hidden'); }
      else { studentView.classList.add('hidden'); }
    }
    if (adminView) {
      if (mode === 'admin') { adminView.classList.remove('hidden'); }
      else { adminView.classList.add('hidden'); }
    }
  }

  if (tabStudent) {
    tabStudent.addEventListener('click', function() { setActiveMode('student'); });
  }
  if (tabAdmin) {
    tabAdmin.addEventListener('click', function() { setActiveMode('admin'); });
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

  // ── 2. Load stages ─────────────────────────────────────────────

  function loadStages() {
    API.get('/flappy/stages').then(function(data) {
      stages = data.stages || {};
      renderStageSelector();
      loadUnlockedStages();
    }).catch(function(err) {
      console.error('Failed to load flappy stages:', err);
    });
  }

  function loadUnlockedStages() {
    var tid = teamId();
    if (!tid) {
      unlockedStages = [1];
      renderStageSelector();
      return;
    }

    API.get('/flappy/unlocked/' + encodeURIComponent(tid)).then(function(data) {
      unlockedStages = data || [1];
      renderStageSelector();
      renderStageResultsGrid();
    }).catch(function(err) {
      console.error('Failed to load unlocked stages:', err);
      unlockedStages = [1];
      renderStageSelector();
    });
  }

  // ── 3. Stage selector ──────────────────────────────────────────

  function renderStageSelector() {
    if (!stageSelect) return;
    var stageIds = Object.keys(stages).map(Number).sort(function(a, b) { return a - b; });
    stageSelect.innerHTML = stageIds.filter(function(sid) {
      return unlockedStages.indexOf(sid) !== -1;
    }).map(function(sid) {
      return '<option value="' + sid + '">Stage ' + sid + ' — ' + (stages[sid].label || '') + '</option>';
    }).join('');
  }

  // ── 4. Deploy (Student Practice) ───────────────────────────────

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
      }).then(function(data) {
        deployBtn.disabled = false;
        deployBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">smart_toy</span> Deploy My Classifier';
        showDeployStatus('', '');

        lastPlayResult = data;
        loadStudentReplay(data);
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

  // ── 5. Submit for Competition (Student) ─────────────────────────

  if (submitCompBtn) {
    submitCompBtn.addEventListener('click', function() {
      var tid = teamId();
      var tname = teamName();
      if (!tid) {
        showSubmitCompStatus('No team ID. Please log in first.', 'error');
        return;
      }
      if (!modelStatus || !modelStatus.has_model) {
        showSubmitCompStatus('No classifier submitted. Go to the Classifier page first.', 'error');
        return;
      }

      submitCompBtn.disabled = true;
      submitCompBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">hourglass_top</span> Submitting...';

      API.post('/flappy/submit-model', {
        team_id: tid,
        team_name: tname,
      }).then(function(data) {
        submitCompBtn.disabled = true;
        submitCompBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">check_circle</span> Submitted';
        showSubmitCompStatus('Model submitted for competition!', 'success');
      }).catch(function(err) {
        submitCompBtn.disabled = false;
        submitCompBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">emoji_events</span> Submit for Competition';
        showSubmitCompStatus('Submission failed: ' + err.message, 'error');
      });
    });
  }

  function showSubmitCompStatus(msg, type) {
    if (!submitCompStatus) return;
    if (!msg) { submitCompStatus.innerHTML = ''; return; }
    var color = type === 'success' ? 'text-emerald-600' : 'text-error';
    var icon = type === 'success' ? 'check_circle' : 'error';
    submitCompStatus.innerHTML = '<div class="flex items-center gap-2 mt-2 ' + color + '">' +
      '<span class="material-symbols-outlined text-[16px]">' + icon + '</span>' +
      '<span class="text-xs font-medium">' + msg + '</span></div>';
  }

  // ── 6. Stage results grid + result popup ────────────────────────

  function renderStageResultsGrid() {
    var grid = document.getElementById('flappy-stage-results-grid');
    if (!grid) return;

    var stageIds = Object.keys(stages).map(Number).sort();
    grid.innerHTML = stageIds.map(function(sid) {
      var r = stageResults[sid];

      if (r) {
        var passedClass = r.passed ? 'bg-emerald-50 border-emerald-200' : 'bg-red-50 border-red-200';
        var passedIcon = r.passed
          ? '<span class="material-symbols-outlined text-emerald-600 text-[14px]">check_circle</span>'
          : '<span class="material-symbols-outlined text-red-500 text-[14px]">cancel</span>';
        var scoreColor = r.passed ? 'text-emerald-700' : 'text-red-600';

        var scoresHtml = '';
        if (r.scores && r.scores.length > 0) {
          scoresHtml = '<div class="flex flex-wrap gap-1 mt-2">' +
            r.scores.map(function(sc) {
              return '<span class="inline-block px-1.5 py-0.5 rounded text-[9px] font-bold ' +
                (sc > 0 ? 'bg-white/80 text-on-surface' : 'bg-red-100 text-red-600') + '">' + sc + '</span>';
            }).join('') +
          '</div>';
        }

        return '<div class="p-3 rounded-xl border ' + passedClass + '">' +
          '<div class="flex items-center gap-2">' +
            passedIcon +
            '<span class="text-xs font-bold text-on-surface flex-1">Stage ' + sid + '</span>' +
            '<span class="text-sm font-black ' + scoreColor + ' tracking-tighter ml-1">' + r.avg_score + '</span>' +
          '</div>' +
          scoresHtml +
        '</div>';
      } else {
        return '<div class="flex items-center gap-3 p-3 rounded-xl border border-surface-container-high bg-surface-container-low">' +
          '<div class="flex items-center gap-2 flex-1 min-w-0">' +
            '<span class="material-symbols-outlined text-on-surface-variant/40 text-[14px]">radio_button_unchecked</span>' +
            '<span class="text-xs font-medium text-on-surface-variant/60 truncate">Stage ' + sid + '</span>' +
          '</div>' +
          '<span class="text-sm text-on-surface-variant/30 font-bold tracking-tighter">—</span>' +
        '</div>';
      }
    }).join('');
  }

  function updateStageResult(data) {
    if (!data || !data.summary) return;
    var sid = data.stage_id;
    var s = data.summary;
    stageResults[sid] = {
      avg_score: s.avg_score,
      max_score: s.max_score,
      scores: s.scores || [],
      passed: s.passed,
    };
    renderStageResultsGrid();
  }

  function showResultPopup(data) {
    var modal = document.getElementById('flappy-result-modal');
    var content = document.getElementById('flappy-result-modal-content');
    var closeBtn = document.getElementById('flappy-result-modal-close');
    if (!modal || !content) return;

    var s = data.summary || {};
    var sid = data.stage_id;
    var stageInfo = stages[sid] || {};
    var physics = data.physics || {};

    var passedBadge = s.passed
      ? '<div class="flex items-center justify-center gap-2 text-emerald-600 bg-emerald-50 rounded-full py-2 px-4 mb-4"><span class="material-symbols-outlined">check_circle</span><span class="font-bold text-sm">Stage ' + sid + ' Passed!</span></div>'
      : '<div class="flex items-center justify-center gap-2 text-red-600 bg-red-50 rounded-full py-2 px-4 mb-4"><span class="material-symbols-outlined">cancel</span><span class="font-bold text-sm">Stage ' + sid + ' Not Passed</span></div>';

    content.innerHTML =
      '<div class="text-center">' +
        '<p class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-2">Stage ' + sid + ' — ' + (stageInfo.label || '') + '</p>' +
        '<p class="text-5xl font-black negative-tracking text-on-surface mb-2">' + s.avg_score + '</p>' +
        '<p class="text-sm text-on-surface-variant mb-4">Score' + (stageInfo.pass_avg ? ' / Pass: ' + stageInfo.pass_avg : '') + '</p>' +
        passedBadge +
        '<div class="grid grid-cols-3 gap-3 text-center">' +
          '<div class="bg-surface-container-low rounded-xl p-3">' +
            '<p class="text-lg font-black text-on-surface">' + (s.max_score != null ? s.max_score : '--') + '</p>' +
            '<p class="text-[9px] text-on-surface-variant font-bold uppercase">Max</p>' +
          '</div>' +
          '<div class="bg-surface-container-low rounded-xl p-3">' +
            '<p class="text-lg font-black text-on-surface">' + (physics.gap_size || '--') + 'px</p>' +
            '<p class="text-[9px] text-on-surface-variant font-bold uppercase">Gap</p>' +
          '</div>' +
          '<div class="bg-surface-container-low rounded-xl p-3">' +
            '<p class="text-lg font-black text-on-surface">' + (physics.features_used || '--') + '/' + (physics.features_available || '--') + '</p>' +
            '<p class="text-[9px] text-on-surface-variant font-bold uppercase">Features</p>' +
          '</div>' +
        '</div>' +
      '</div>';

    modal.classList.remove('hidden');

    function closeModal() {
      modal.classList.add('hidden');
      if (closeBtn) closeBtn.removeEventListener('click', closeModal);
    }
    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', function(e) {
      if (e.target === modal) closeModal();
    });
  }

  // ══════════════════════════════════════════════════════════════════
  //  STUDENT CANVAS REPLAY
  // ══════════════════════════════════════════════════════════════════

  function resizeStudentCanvas() {
    if (!canvasContainer || !canvas) return;
    var w = canvasContainer.clientWidth;
    var h = canvasContainer.clientHeight || Math.round(w * 3 / 5);
    canvas.width = w;
    canvas.height = h;
    CANVAS_W = w;
    CANVAS_H = h;
    SCALE_X = CANVAS_W / WORLD_W;
    SCALE_Y = CANVAS_H / WORLD_H;
    renderStudentFrame();
  }

  function loadStudentReplay(replay) {
    replayData = replay;
    episodeIdx = 0;
    frameIdx = 0;
    playing = true;
    holdCount = 0;

    WORLD_W = replay.world_width || 420;
    WORLD_H = replay.world_height || 580;
    GROUND_H = replay.ground_height || 60;

    resizeStudentCanvas();
    if (btnPlay) btnPlay.textContent = 'Pause';
    renderStudentFrame();

    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = requestAnimationFrame(studentTick);
  }

  function currentEpisode() {
    if (!replayData || !replayData.episodes) return null;
    return replayData.episodes[episodeIdx] || null;
  }

  // ── Shared drawing primitives ───────────────────────────────────

  function drawBackground(c, cw, ch) {
    var grd = c.createLinearGradient(0, 0, 0, ch);
    grd.addColorStop(0, '#0e1117');
    grd.addColorStop(1, '#1a1f2e');
    c.fillStyle = grd;
    c.fillRect(0, 0, cw, ch);
  }

  function drawGround(c, cw, ch, scY, groundH) {
    var gy = ch - (groundH * scY);
    c.fillStyle = '#1e2533';
    c.fillRect(0, gy, cw, ch - gy);
    c.strokeStyle = 'rgba(255,255,255,0.06)';
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(0, gy);
    c.lineTo(cw, gy);
    c.stroke();
  }

  function drawPipeAtScreenX(c, screenX, gapY, gapSize, scX, scY, ch) {
    var sw = PIPE_WIDTH * scX;
    var sgapY = gapY * scY;
    var sgapSize = gapSize * scY;
    var capH = 8;

    var topBottom = sgapY - sgapSize / 2;
    var botTop = sgapY + sgapSize / 2;

    // Top pipe
    c.fillStyle = '#22c55e';
    c.fillRect(screenX, 0, sw, topBottom);
    c.fillStyle = '#15803d';
    c.fillRect(screenX - 3, topBottom - capH, sw + 6, capH);

    // Bottom pipe
    c.fillStyle = '#22c55e';
    c.fillRect(screenX, botTop, sw, ch - botTop);
    c.fillStyle = '#15803d';
    c.fillRect(screenX - 3, botTop, sw + 6, capH);
  }

  function drawPipes(c, pipeMap, cameraX, worldW, scX, scY, cw, ch) {
    var viewLeft = cameraX - 50;
    var viewRight = cameraX + worldW + 50;
    for (var i = 0; i < pipeMap.length; i++) {
      var pipe = pipeMap[i];
      if (pipe.world_x >= viewLeft && pipe.world_x <= viewRight) {
        var pipeScreenX = (pipe.world_x - cameraX) * scX;
        drawPipeAtScreenX(c, pipeScreenX, pipe.gap_y, pipe.gap_size, scX, scY, ch);
      }
    }
  }

  function drawBird(c, screenX, screenY, alive, score, color, name) {
    var radius = 12;

    c.save();
    c.globalAlpha = alive ? 1.0 : 0.3;

    // Body
    c.beginPath();
    c.arc(screenX, screenY, radius, 0, Math.PI * 2);
    c.fillStyle = color;
    c.fill();
    c.strokeStyle = 'rgba(0,0,0,0.3)';
    c.lineWidth = 1.5;
    c.stroke();

    // Eye
    c.beginPath();
    c.arc(screenX + 4, screenY - 3, 4, 0, Math.PI * 2);
    c.fillStyle = '#ffffff';
    c.fill();
    c.beginPath();
    c.arc(screenX + 5, screenY - 3, 2, 0, Math.PI * 2);
    c.fillStyle = '#0f172a';
    c.fill();

    // Dead marker
    if (!alive) {
      c.strokeStyle = '#ef4444';
      c.lineWidth = 3;
      c.lineCap = 'round';
      var xOff = 10;
      c.beginPath();
      c.moveTo(screenX - xOff, screenY - xOff);
      c.lineTo(screenX + xOff, screenY + xOff);
      c.stroke();
      c.beginPath();
      c.moveTo(screenX + xOff, screenY - xOff);
      c.lineTo(screenX - xOff, screenY + xOff);
      c.stroke();
    }

    // Team name above
    c.font = '600 10px Inter, sans-serif';
    c.textAlign = 'center';
    c.fillStyle = color;
    c.fillText(name || '', screenX, screenY - radius - 14);

    // Score below
    c.font = '500 10px Inter, sans-serif';
    c.fillStyle = 'rgba(255,255,255,0.7)';
    c.fillText('Score: ' + score, screenX, screenY + radius + 14);

    c.restore();
  }

  // ── Student frame render ────────────────────────────────────────

  function renderStudentFrame() {
    if (!replayData || !ctx) return;

    var ep = currentEpisode();
    if (!ep || !ep.frames || ep.frames.length === 0) return;

    var frame = ep.frames[frameIdx];
    if (!frame) return;

    var birdWorldX = frame.bird_x || 0;
    var birdWorldY = frame.bird_y || 0;

    var birdScreenX = CANVAS_W * BIRD_SCREEN_X_RATIO;
    var cameraX = birdWorldX - (birdScreenX / SCALE_X);

    drawBackground(ctx, CANVAS_W, CANVAS_H);

    var pipeMap = ep.pipe_map || [];
    drawPipes(ctx, pipeMap, cameraX, WORLD_W, SCALE_X, SCALE_Y, CANVAS_W, CANVAS_H);

    drawGround(ctx, CANVAS_W, CANVAS_H, SCALE_Y, GROUND_H);

    var birdScreenY = birdWorldY * SCALE_Y;
    drawBird(ctx, birdScreenX, birdScreenY, frame.alive, frame.score, TEAM_COLORS[0], teamName());

    // Update overlay
    var stageId = replayData.stage_id != null ? replayData.stage_id : '--';
    if (overlayStage) overlayStage.textContent = stageId;
    if (overlayScore) overlayScore.textContent = frame.score;
  }

  function studentTick() {
    animFrameId = requestAnimationFrame(studentTick);

    if (!playing || !replayData) return;

    frameTick++;
    if (frameTick < FRAME_INTERVAL) return;
    frameTick = 0;

    var ep = currentEpisode();
    if (!ep || !ep.frames) return;

    if (frameIdx < ep.frames.length - 1) {
      frameIdx++;
    } else {
      // Episode finished
      if (episodeIdx < replayData.episodes.length - 1) {
        holdCount++;
        if (holdCount >= 30) {
          holdCount = 0;
          episodeIdx++;
          frameIdx = 0;
        }
      } else {
        // All episodes done
        playing = false;
        if (btnPlay) btnPlay.textContent = 'Play';
        if (lastPlayResult) {
          updateStageResult(lastPlayResult);
          showResultPopup(lastPlayResult);
        }
      }
    }

    renderStudentFrame();
  }

  // Play/Pause (student)
  if (btnPlay) {
    btnPlay.addEventListener('click', function() {
      playing = !playing;
      btnPlay.textContent = playing ? 'Pause' : 'Play';
      if (playing) renderStudentFrame();
    });
  }

  // Next Episode (student)
  if (btnNext) {
    btnNext.addEventListener('click', function() {
      if (!replayData) return;
      holdCount = 0;
      if (episodeIdx < replayData.episodes.length - 1) {
        episodeIdx++;
      } else {
        episodeIdx = 0;
      }
      frameIdx = 0;
      renderStudentFrame();
    });
  }

  // Idle canvas (student)
  function drawIdleCanvas() {
    if (!ctx) return;
    resizeStudentCanvas();
    drawBackground(ctx, CANVAS_W, CANVAS_H);

    ctx.fillStyle = '#64748b';
    ctx.font = '500 14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Deploy your classifier to see replay here', CANVAS_W / 2, CANVAS_H / 2 - 10);
    ctx.font = '400 12px Inter, sans-serif';
    ctx.fillStyle = '#475569';
    ctx.fillText('Select a stage and click Deploy', CANVAS_W / 2, CANVAS_H / 2 + 14);
  }

  // ══════════════════════════════════════════════════════════════════
  //  ADMIN MODE
  // ══════════════════════════════════════════════════════════════════

  // ── Admin Login ─────────────────────────────────────────────────

  if (adminLoginBtn) {
    adminLoginBtn.addEventListener('click', function() {
      var pw = adminPasswordInput ? adminPasswordInput.value.trim() : '';
      if (!pw) {
        if (adminLoginError) adminLoginError.textContent = 'Please enter the admin password.';
        return;
      }
      adminPw = pw;
      if (adminLoginError) adminLoginError.textContent = '';
      if (adminLoginPanel) adminLoginPanel.classList.add('hidden');
      if (adminCompPanel) adminCompPanel.classList.remove('hidden');
      fetchSubmittedTeams();
    });
  }

  // ── Submitted Teams ─────────────────────────────────────────────

  function fetchSubmittedTeams() {
    API.get('/flappy/submitted-teams').then(function(teams) {
      renderSubmittedTeams(teams);
    }).catch(function(err) {
      console.error('Failed to fetch submitted teams:', err);
      if (adminTeamsList) {
        adminTeamsList.innerHTML = '<p class="text-sm text-error">Failed to load teams: ' + err.message + '</p>';
      }
    });
  }

  function renderSubmittedTeams(teams) {
    if (!adminTeamsList) return;
    if (!teams || teams.length === 0) {
      adminTeamsList.innerHTML = '<p class="text-sm text-on-surface-variant">No teams have submitted yet.</p>';
      return;
    }

    adminTeamsList.innerHTML = teams.map(function(t, i) {
      var color = TEAM_COLORS[i % TEAM_COLORS.length];
      var initial = ((t.team_name || '?').charAt(0)).toUpperCase();
      return '<div class="flex items-center gap-3 p-3 rounded-xl border border-surface-container-high bg-surface-container-low">' +
        '<div class="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-xs flex-shrink-0" style="background:' + color + '">' + initial + '</div>' +
        '<div class="flex-1 min-w-0">' +
          '<p class="text-sm font-bold text-on-surface truncate">' + (t.team_name || t.team_id) + '</p>' +
          '<p class="text-[10px] text-on-surface-variant">' + (t.team_id || '') + '</p>' +
        '</div>' +
      '</div>';
    }).join('');
  }

  if (adminRefreshTeams) {
    adminRefreshTeams.addEventListener('click', function() {
      fetchSubmittedTeams();
    });
  }

  // ── Start Competition ───────────────────────────────────────────

  if (adminStartBtn) {
    adminStartBtn.addEventListener('click', function() {
      adminStartBtn.disabled = true;
      adminStartBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">hourglass_top</span> Running Competition...';
      if (adminStartStatus) adminStartStatus.innerHTML = '';

      API.post('/flappy/competition', {
        admin_password: adminPw,
      }).then(function(data) {
        adminStartBtn.disabled = false;
        adminStartBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">flag</span> Start Competition';

        competitionResult = data;
        adminStageIdx = 0;
        adminEliminationList = [];

        if (adminStartStatus) {
          adminStartStatus.innerHTML = '<div class="flex items-center gap-2 mt-2 text-emerald-600">' +
            '<span class="material-symbols-outlined text-[16px]">check_circle</span>' +
            '<span class="text-xs font-medium">Competition complete! ' + data.total_teams + ' teams, ' + data.stages_played + ' stages. Replay loaded.</span>' +
          '</div>';
        }

        // Clear previous state
        if (adminEliminationEl) adminEliminationEl.innerHTML = '';
        if (adminFinalRanking) adminFinalRanking.innerHTML = '';

        // Start replay from stage 1
        startAdminStageReplay();
      }).catch(function(err) {
        adminStartBtn.disabled = false;
        adminStartBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">flag</span> Start Competition';
        if (adminStartStatus) {
          adminStartStatus.innerHTML = '<div class="flex items-center gap-2 mt-2 text-error">' +
            '<span class="material-symbols-outlined text-[16px]">error</span>' +
            '<span class="text-xs font-medium">Competition failed: ' + err.message + '</span>' +
          '</div>';
        }
      });
    });
  }

  // ══════════════════════════════════════════════════════════════════
  //  ADMIN CANVAS REPLAY (Multi-Bird)
  // ══════════════════════════════════════════════════════════════════

  function resizeAdminCanvas() {
    if (!adminCanvasContainer || !adminCanvas) return;
    var w = adminCanvasContainer.clientWidth;
    var h = adminCanvasContainer.clientHeight || Math.round(w * 3 / 5);
    adminCanvas.width = w;
    adminCanvas.height = h;
    ADMIN_CANVAS_W = w;
    ADMIN_CANVAS_H = h;
    ADMIN_SCALE_X = ADMIN_CANVAS_W / WORLD_W;
    ADMIN_SCALE_Y = ADMIN_CANVAS_H / WORLD_H;
    renderAdminFrame();
  }

  function getCurrentAdminStage() {
    if (!competitionResult || !competitionResult.stage_results) return null;
    return competitionResult.stage_results[adminStageIdx] || null;
  }

  function getCurrentAdminReplay() {
    if (!competitionResult || !competitionResult.replay) return null;
    var stageInfo = getCurrentAdminStage();
    if (!stageInfo) return null;
    var stageKey = String(stageInfo.stage_id);
    return competitionResult.replay[stageKey] || null;
  }

  function startAdminStageReplay() {
    var replay = getCurrentAdminReplay();
    if (!replay || !replay.frames || replay.frames.length === 0) return;

    adminFrameIdx = 0;
    adminPlaying = true;
    adminHoldCount = 0;

    // Read world dims from replay if available
    if (replay.world_width) WORLD_W = replay.world_width;
    if (replay.world_height) WORLD_H = replay.world_height;
    if (replay.ground_height) GROUND_H = replay.ground_height;

    resizeAdminCanvas();
    if (adminBtnPlay) adminBtnPlay.textContent = 'Pause';
    if (adminBtnNextStage) adminBtnNextStage.style.display = 'none';
    renderAdminFrame();

    if (adminAnimFrameId) cancelAnimationFrame(adminAnimFrameId);
    adminAnimFrameId = requestAnimationFrame(adminTick);
  }

  // Build a team->color map for consistent coloring
  function buildTeamColorMap() {
    if (!competitionResult || !competitionResult.final_ranking) return {};
    var map = {};
    for (var i = 0; i < competitionResult.final_ranking.length; i++) {
      map[competitionResult.final_ranking[i].team_id] = TEAM_COLORS[i % TEAM_COLORS.length];
    }
    return map;
  }

  function renderAdminFrame() {
    if (!adminCtx || !competitionResult) return;

    var replay = getCurrentAdminReplay();
    if (!replay || !replay.frames || replay.frames.length === 0) return;

    var frame = replay.frames[adminFrameIdx];
    if (!frame) return;

    var birds = frame.birds || [];
    var pipeMap = replay.pipe_map || [];
    var gapSize = replay.gap_size || 130;

    // Find leading alive bird for camera
    var leadingX = frame.bird_x || 0;
    var leadingScore = 0;
    var aliveCount = 0;
    for (var b = 0; b < birds.length; b++) {
      if (birds[b].alive) {
        aliveCount++;
        if (birds[b].score > leadingScore) {
          leadingScore = birds[b].score;
        }
      }
    }

    var birdScreenX = ADMIN_CANVAS_W * BIRD_SCREEN_X_RATIO;
    var cameraX = leadingX - (birdScreenX / ADMIN_SCALE_X);

    drawBackground(adminCtx, ADMIN_CANVAS_W, ADMIN_CANVAS_H);
    drawPipes(adminCtx, pipeMap, cameraX, WORLD_W, ADMIN_SCALE_X, ADMIN_SCALE_Y, ADMIN_CANVAS_W, ADMIN_CANVAS_H);
    drawGround(adminCtx, ADMIN_CANVAS_W, ADMIN_CANVAS_H, ADMIN_SCALE_Y, GROUND_H);

    // Draw each bird
    var colorMap = buildTeamColorMap();
    for (var i = 0; i < birds.length; i++) {
      var bird = birds[i];
      var color = colorMap[bird.team_id] || TEAM_COLORS[i % TEAM_COLORS.length];
      var birdY = bird.y * ADMIN_SCALE_Y;
      drawBird(adminCtx, birdScreenX, birdY, bird.alive, bird.score, color, bird.team_name);
    }

    // Overlay: stage, alive count, leading score
    var stageInfo = getCurrentAdminStage();
    if (adminInfoStage) adminInfoStage.textContent = stageInfo ? stageInfo.stage_id : '--';
    if (adminInfoAlive) adminInfoAlive.textContent = aliveCount + ' / ' + birds.length;
    if (adminInfoScore) adminInfoScore.textContent = leadingScore;

    // Stage + alive + score on canvas
    adminCtx.save();
    adminCtx.font = '700 14px Inter, sans-serif';
    adminCtx.textAlign = 'left';
    adminCtx.fillStyle = 'rgba(255,255,255,0.8)';
    adminCtx.fillText('Stage ' + (stageInfo ? stageInfo.stage_id : '?'), 16, 28);
    adminCtx.font = '500 12px Inter, sans-serif';
    adminCtx.fillStyle = 'rgba(255,255,255,0.6)';
    adminCtx.fillText('Alive: ' + aliveCount + '/' + birds.length + '   |   Top Score: ' + leadingScore, 16, 48);
    adminCtx.restore();
  }

  // Track which birds have been added to elimination list
  var eliminatedSet = {};

  function updateEliminationDuringReplay(frame) {
    if (!frame || !frame.birds) return;
    var stageInfo = getCurrentAdminStage();
    var stageId = stageInfo ? stageInfo.stage_id : '?';
    var colorMap = buildTeamColorMap();

    for (var i = 0; i < frame.birds.length; i++) {
      var bird = frame.birds[i];
      var key = bird.team_id + '_' + stageId;
      if (!bird.alive && !eliminatedSet[key]) {
        eliminatedSet[key] = true;
        adminEliminationList.unshift({
          team_name: bird.team_name,
          team_id: bird.team_id,
          score: bird.score,
          stage: stageId,
          color: colorMap[bird.team_id] || TEAM_COLORS[i % TEAM_COLORS.length],
        });
        renderEliminationList();
      }
    }
  }

  function renderEliminationList() {
    if (!adminEliminationEl) return;
    if (adminEliminationList.length === 0) {
      adminEliminationEl.innerHTML = '<p class="text-sm text-on-surface-variant/50 text-center py-4">No eliminations yet</p>';
      return;
    }

    adminEliminationEl.innerHTML = adminEliminationList.map(function(e, i) {
      var rank = adminEliminationList.length - i;
      return '<div class="flex items-center gap-3 p-2 rounded-lg border border-surface-container-high">' +
        '<span class="text-xs font-black text-on-surface-variant/40 w-5 text-right">' + rank + '</span>' +
        '<div class="w-5 h-5 rounded-full flex-shrink-0" style="background:' + e.color + '"></div>' +
        '<span class="text-xs font-bold text-on-surface flex-1 truncate">' + e.team_name + '</span>' +
        '<span class="text-xs text-on-surface-variant">S' + e.stage + '</span>' +
        '<span class="text-xs font-bold text-on-surface">' + e.score + '</span>' +
      '</div>';
    }).join('');
  }

  function adminTick() {
    adminAnimFrameId = requestAnimationFrame(adminTick);

    if (!adminPlaying || !competitionResult) return;

    adminFrameTick++;
    if (adminFrameTick < ADMIN_FRAME_INTERVAL) return;
    adminFrameTick = 0;

    var replay = getCurrentAdminReplay();
    if (!replay || !replay.frames) return;

    if (adminFrameIdx < replay.frames.length - 1) {
      adminFrameIdx++;
      var frame = replay.frames[adminFrameIdx];
      updateEliminationDuringReplay(frame);
    } else {
      // Stage replay finished
      adminPlaying = false;
      if (adminBtnPlay) adminBtnPlay.textContent = 'Play';

      // Check if more stages
      if (adminStageIdx < competitionResult.stage_results.length - 1) {
        if (adminBtnNextStage) {
          adminBtnNextStage.style.display = '';
          adminBtnNextStage.textContent = 'Next Stage';
        }
      } else {
        // All stages done — show final ranking
        renderFinalRanking();
        if (adminBtnNextStage) adminBtnNextStage.style.display = 'none';
      }
    }

    renderAdminFrame();
  }

  // Admin Play/Pause
  if (adminBtnPlay) {
    adminBtnPlay.addEventListener('click', function() {
      adminPlaying = !adminPlaying;
      adminBtnPlay.textContent = adminPlaying ? 'Pause' : 'Play';
      if (adminPlaying) renderAdminFrame();
    });
  }

  // Admin Next Stage
  if (adminBtnNextStage) {
    adminBtnNextStage.addEventListener('click', function() {
      if (!competitionResult) return;
      if (adminStageIdx < competitionResult.stage_results.length - 1) {
        adminStageIdx++;
        startAdminStageReplay();
      }
    });
  }

  // ── Final Ranking ───────────────────────────────────────────────

  function renderFinalRanking() {
    if (!adminFinalRanking || !competitionResult || !competitionResult.final_ranking) return;

    var ranking = competitionResult.final_ranking;
    var medals = ['', '', ''];

    adminFinalRanking.innerHTML =
      '<p class="text-sm font-bold text-on-surface mb-3 uppercase tracking-wider">Final Ranking</p>' +
      ranking.map(function(r, i) {
        var rank = i + 1;
        var color = TEAM_COLORS[i % TEAM_COLORS.length];
        var initial = ((r.team_name || '?').charAt(0)).toUpperCase();
        var medalHtml = '';
        if (rank === 1) medalHtml = '<span class="text-lg">&#x1F947;</span>';
        else if (rank === 2) medalHtml = '<span class="text-lg">&#x1F948;</span>';
        else if (rank === 3) medalHtml = '<span class="text-lg">&#x1F949;</span>';

        var elimText = r.eliminated_stage ? 'Eliminated S' + r.eliminated_stage : 'Survived';
        var elimClass = r.eliminated_stage ? 'text-red-500' : 'text-emerald-600';

        return '<div class="flex items-center gap-3 p-3 rounded-xl border ' +
          (rank <= 3 ? 'border-amber-200 bg-amber-50/50' : 'border-surface-container-high bg-surface-container-low') + '">' +
          '<span class="text-lg font-black w-6 text-center ' + (rank <= 3 ? 'text-amber-500' : 'text-on-surface-variant/40') + '">' + rank + '</span>' +
          medalHtml +
          '<div class="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-xs flex-shrink-0" style="background:' + color + '">' + initial + '</div>' +
          '<div class="flex-1 min-w-0">' +
            '<p class="text-sm font-bold text-on-surface truncate">' + (r.team_name || r.team_id) + '</p>' +
            '<p class="text-[10px] ' + elimClass + ' font-medium">' + elimText + '</p>' +
          '</div>' +
          '<div class="text-right">' +
            '<p class="text-lg font-black text-on-surface tracking-tighter">' + (r.final_score != null ? r.final_score : '--') + '</p>' +
            '<p class="text-[9px] text-on-surface-variant font-bold uppercase">Score</p>' +
          '</div>' +
        '</div>';
      }).join('');
  }

  // Idle canvas (admin)
  function drawAdminIdleCanvas() {
    if (!adminCtx) return;
    resizeAdminCanvas();
    drawBackground(adminCtx, ADMIN_CANVAS_W, ADMIN_CANVAS_H);

    adminCtx.fillStyle = '#64748b';
    adminCtx.font = '500 14px Inter, sans-serif';
    adminCtx.textAlign = 'center';
    adminCtx.fillText('Start a competition to see replay here', ADMIN_CANVAS_W / 2, ADMIN_CANVAS_H / 2 - 10);
    adminCtx.font = '400 12px Inter, sans-serif';
    adminCtx.fillStyle = '#475569';
    adminCtx.fillText('Submit teams, then click Start Competition', ADMIN_CANVAS_W / 2, ADMIN_CANVAS_H / 2 + 14);
  }

  // ── Resize handling ─────────────────────────────────────────────

  window.addEventListener('resize', function() {
    resizeStudentCanvas();
    resizeAdminCanvas();
  });

  // ── Cleanup ─────────────────────────────────────────────────────

  var observer = new MutationObserver(function() {
    if (!document.getElementById('replay-canvas')) {
      if (animFrameId) cancelAnimationFrame(animFrameId);
      if (adminAnimFrameId) cancelAnimationFrame(adminAnimFrameId);
      observer.disconnect();
    }
  });
  var pageContent = document.getElementById('page-content');
  if (pageContent) {
    observer.observe(pageContent, { childList: true });
  }

  // ── Init ────────────────────────────────────────────────────────
  setActiveMode('student');
  drawIdleCanvas();
  if (adminCtx) drawAdminIdleCanvas();
  checkModelStatus();
  loadStages();
}
