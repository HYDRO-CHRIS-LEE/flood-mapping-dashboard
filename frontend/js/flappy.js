/* Flappy Bird Competition page logic */

function init_flappy() {
  // ── State ───────────────────────────────────────────────────────
  var stages = {};
  var architectures = [];
  var unlockedStages = [];
  var currentModelId = null;
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
  var teamColorMap = {};

  // ── DOM refs ────────────────────────────────────────────────────
  var canvas = document.getElementById('replay-canvas');
  var ctx = canvas ? canvas.getContext('2d') : null;
  var canvasContainer = document.getElementById('replay-container');

  var stageBadgesRow = document.getElementById('flappy-stage-badges');
  var archSelector = document.getElementById('flappy-arch-select');
  var hyperparamInfo = document.getElementById('flappy-hyperparam-info');
  var uploadStateDict = document.getElementById('flappy-upload-sd');
  var uploadMetadata = document.getElementById('flappy-upload-meta');
  var uploadBtn = document.getElementById('flappy-upload-btn');
  var uploadStatus = document.getElementById('flappy-upload-status');
  var submitStageSelect = document.getElementById('flappy-submit-stage');
  var submitBtn = document.getElementById('flappy-submit-btn');
  var submitStatus = document.getElementById('flappy-submit-status');

  var adminPassword = document.getElementById('flappy-admin-pw');
  var adminStageSelect = document.getElementById('flappy-admin-stage');
  var raceBtn = document.getElementById('flappy-race-btn');
  var raceStatus = document.getElementById('flappy-race-status');

  var btnPlay = document.getElementById('flappy-btn-play');
  var btnNext = document.getElementById('flappy-btn-next');
  var overlayStage = document.getElementById('flappy-info-stage');
  var overlayGap = document.getElementById('flappy-info-gap');
  var overlayEpisode = document.getElementById('flappy-info-episode');
  var overlayFrame = document.getElementById('flappy-info-frame');
  var overlayAlive = document.getElementById('flappy-info-alive');

  var lbTabsContainer = document.getElementById('flappy-lb-tabs');
  var lbBody = document.getElementById('flappy-lb-body');

  if (!canvas || !ctx) return;

  // ── Helpers ─────────────────────────────────────────────────────

  function teamName() {
    return localStorage.getItem('earthai_name') || 'Student';
  }

  function teamId() {
    return localStorage.getItem('earthai_team_id') || '';
  }

  // ── 1. Load stages and architectures ────────────────────────────

  function loadStagesAndArchitectures() {
    API.get('/flappy/stages').then(function(data) {
      stages = data.stages || {};
      architectures = data.architectures || [];
      renderArchitectureSelector();
      renderStageSelectors();
      loadUnlockedStages();
    }).catch(function(err) {
      console.error('Failed to load flappy stages:', err);
    });
  }

  function loadUnlockedStages() {
    var name = teamName();
    API.get('/flappy/unlocked/' + encodeURIComponent(name)).then(function(data) {
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

  // ── 2. Stage badges ─────────────────────────────────────────────

  function renderStageBadges() {
    if (!stageBadgesRow) return;
    var stageIds = Object.keys(stages).map(Number).sort(function(a, b) { return a - b; });
    stageBadgesRow.innerHTML = stageIds.map(function(sid) {
      var s = stages[sid];
      var isUnlocked = unlockedStages.indexOf(sid) !== -1;
      var isPassed = unlockedStages.indexOf(sid + 1) !== -1;
      // Stage 5 has no "next" stage, so passed = in unlocked and we'd need leaderboard data
      // For simplicity: if the next stage is unlocked, this one is passed

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

  // ── 3. Architecture selector ────────────────────────────────────

  function renderArchitectureSelector() {
    if (!archSelector) return;
    archSelector.innerHTML = architectures.map(function(arch) {
      return '<option value="' + arch + '">' + arch + '</option>';
    }).join('');
    updateHyperparamInfo();
  }

  function updateHyperparamInfo() {
    if (!hyperparamInfo || !archSelector) return;
    var arch = archSelector.value;
    var info = {
      model1: { layers: '4 -> 64 -> 64 -> 2', activation: 'ReLU', params: '~4,674' },
      model2: { layers: '4 -> 128 -> 128 -> 2', activation: 'ReLU', params: '~17,538' },
      model3: { layers: '4 -> 256 -> 128 -> 64 -> 2', activation: 'ReLU', params: '~42,114' },
    };
    var d = info[arch] || { layers: '--', activation: '--', params: '--' };
    hyperparamInfo.innerHTML =
      '<div class="flex justify-between py-2 border-b border-outline-variant/30">' +
        '<span class="text-xs text-on-surface-variant">Layers</span>' +
        '<span class="text-xs font-mono font-semibold text-on-surface">' + d.layers + '</span>' +
      '</div>' +
      '<div class="flex justify-between py-2 border-b border-outline-variant/30">' +
        '<span class="text-xs text-on-surface-variant">Activation</span>' +
        '<span class="text-xs font-mono font-semibold text-on-surface">' + d.activation + '</span>' +
      '</div>' +
      '<div class="flex justify-between py-2">' +
        '<span class="text-xs text-on-surface-variant">Parameters</span>' +
        '<span class="text-xs font-mono font-semibold text-on-surface">' + d.params + '</span>' +
      '</div>';
  }

  if (archSelector) {
    archSelector.addEventListener('change', updateHyperparamInfo);
  }

  // ── Stage selectors ─────────────────────────────────────────────

  function renderStageSelectors() {
    var stageIds = Object.keys(stages).map(Number).sort(function(a, b) { return a - b; });

    // Submit stage select — only unlocked
    if (submitStageSelect) {
      submitStageSelect.innerHTML = stageIds.filter(function(sid) {
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

  // ── 4. Upload ───────────────────────────────────────────────────

  if (uploadBtn) {
    uploadBtn.addEventListener('click', function() {
      if (!uploadStateDict || !uploadMetadata) return;

      var sdFile = uploadStateDict.files && uploadStateDict.files[0];
      var metaFile = uploadMetadata.files && uploadMetadata.files[0];

      if (!sdFile || !metaFile) {
        showUploadStatus('Please select both state_dict.pt and metadata.json files.', 'error');
        return;
      }

      var formData = new FormData();
      formData.append('state_dict', sdFile);
      formData.append('metadata', metaFile);

      uploadBtn.disabled = true;
      uploadBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">hourglass_top</span> Validating...';

      fetch('/api/flappy/upload', {
        method: 'POST',
        headers: {
          'X-Team-Id': teamId(),
          'X-Team-Name': teamName(),
        },
        body: formData,
      })
      .then(function(resp) { return resp.json(); })
      .then(function(json) {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">upload_file</span> Upload & Validate';
        if (json.ok) {
          currentModelId = json.data.model_id;
          showUploadStatus('Model validated successfully. Ready to submit.', 'success');
        } else {
          showUploadStatus(json.message || json.error || 'Validation failed.', 'error');
        }
      })
      .catch(function(err) {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">upload_file</span> Upload & Validate';
        showUploadStatus('Upload failed: ' + err.message, 'error');
      });
    });
  }

  function showUploadStatus(msg, type) {
    if (!uploadStatus) return;
    var color = type === 'success' ? 'text-emerald-600' : 'text-error';
    var icon = type === 'success' ? 'check_circle' : 'error';
    uploadStatus.innerHTML = '<div class="flex items-center gap-2 mt-2 ' + color + '">' +
      '<span class="material-symbols-outlined text-[16px]">' + icon + '</span>' +
      '<span class="text-xs font-medium">' + msg + '</span></div>';
  }

  // ── 5. Submit ───────────────────────────────────────────────────

  if (submitBtn) {
    submitBtn.addEventListener('click', function() {
      if (!currentModelId) {
        showSubmitStatus('Upload and validate a model first.', 'error');
        return;
      }
      if (!submitStageSelect) return;
      var stageId = parseInt(submitStageSelect.value);

      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">hourglass_top</span> Submitting...';

      API.post('/flappy/submit', {
        team_name: teamName(),
        model_id: currentModelId,
        stage_id: stageId,
      }).then(function(data) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">send</span> Submit to Stage';
        showSubmitStatus('Submission saved for Stage ' + stageId + '.', 'success');
        currentModelId = null;
      }).catch(function(err) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px]">send</span> Submit to Stage';
        showSubmitStatus('Submission failed: ' + err.message, 'error');
      });
    });
  }

  function showSubmitStatus(msg, type) {
    if (!submitStatus) return;
    var color = type === 'success' ? 'text-emerald-600' : 'text-error';
    var icon = type === 'success' ? 'check_circle' : 'error';
    submitStatus.innerHTML = '<div class="flex items-center gap-2 mt-2 ' + color + '">' +
      '<span class="material-symbols-outlined text-[16px]">' + icon + '</span>' +
      '<span class="text-xs font-medium">' + msg + '</span></div>';
  }

  // ── 6. Admin Race ───────────────────────────────────────────────

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

  // ── 7. Canvas Replay Viewer ─────────────────────────────────────

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

    // Build team color map
    teamColorMap = {};
    for (var ei = 0; ei < replay.episodes.length; ei++) {
      var ep = replay.episodes[ei];
      if (ep.frames && ep.frames.length > 0) {
        var birds = ep.frames[0].birds || [];
        for (var bi = 0; bi < birds.length; bi++) {
          var name = birds[bi].team_name;
          if (!(name in teamColorMap)) {
            teamColorMap[name] = TEAM_COLORS[Object.keys(teamColorMap).length % TEAM_COLORS.length];
          }
        }
      }
    }

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

  function drawPipe(px, gapY, gapSize) {
    var sx = px * SCALE_X;
    var sw = PIPE_WIDTH * SCALE_X;
    var sgapY = gapY * SCALE_Y;
    var sgapSize = gapSize * SCALE_Y;
    var capH = 8;

    var topPipeBottom = sgapY - sgapSize / 2;
    var bottomPipeTop = sgapY + sgapSize / 2;

    // Top pipe body
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(sx, 0, sw, topPipeBottom);
    // Top pipe cap
    ctx.fillStyle = '#15803d';
    ctx.fillRect(sx - 3, topPipeBottom - capH, sw + 6, capH);

    // Bottom pipe body
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(sx, bottomPipeTop, sw, CANVAS_H - bottomPipeTop);
    // Bottom pipe cap
    ctx.fillStyle = '#15803d';
    ctx.fillRect(sx - 3, bottomPipeTop, sw + 6, capH);
  }

  function drawBirdAtX(bird, worldX, showCrown, totalTeams) {
    var bx = worldX * SCALE_X;
    var by = bird.y * SCALE_Y;
    var radius = 12;
    var color = teamColorMap[bird.team_name] || '#ffffff';
    var alpha = bird.alive ? 1.0 : 0.4;

    ctx.save();
    ctx.globalAlpha = alpha;

    // Body
    ctx.beginPath();
    ctx.arc(bx, by, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Eye
    ctx.beginPath();
    ctx.arc(bx + 4, by - 3, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(bx + 5, by - 3, 2, 0, Math.PI * 2);
    ctx.fillStyle = '#0f172a';
    ctx.fill();

    // Dead marker
    if (!bird.alive) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      var xOff = 10;
      ctx.beginPath();
      ctx.moveTo(bx - xOff, by - xOff);
      ctx.lineTo(bx + xOff, by + xOff);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(bx + xOff, by - xOff);
      ctx.lineTo(bx - xOff, by + xOff);
      ctx.stroke();
    }

    // Team name
    ctx.font = '600 10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = color;
    ctx.fillText(bird.team_name, bx, by - radius - 14);

    // Crown
    if (showCrown) {
      ctx.font = '14px sans-serif';
      ctx.fillText('\uD83D\uDC51', bx, by - radius - 26);
    }

    // Score
    ctx.font = '500 10px Inter, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.fillText('Score: ' + bird.score, bx, by + radius + 14);

    ctx.restore();
  }

  function renderFrame() {
    if (!replayData || !ctx) return;

    var ep = currentEpisode();
    if (!ep || !ep.frames || ep.frames.length === 0) return;

    var frame = ep.frames[frameIdx];
    if (!frame) return;

    drawBackground();

    // Pipes
    var pipes = frame.pipes || [];
    for (var pi = 0; pi < pipes.length; pi++) {
      var p = pipes[pi];
      drawPipe(p.x, p.gap_y, p.gap_size);
    }

    drawGround();

    // Birds
    var birds = frame.birds || [];
    var aliveCount = 0;
    var totalTeams = birds.length;
    for (var ai = 0; ai < birds.length; ai++) {
      if (birds[ai].alive) aliveCount++;
    }
    var isLastSurvivor = (aliveCount === 1 && totalTeams > 1);

    var birdSpacing = (totalTeams > 1) ? Math.min(30, 120 / (totalTeams - 1)) : 0;
    var birdBaseX = 60;
    for (var bi = 0; bi < birds.length; bi++) {
      var b = birds[bi];
      var offsetX = (bi - (totalTeams - 1) / 2) * birdSpacing;
      drawBirdAtX(b, birdBaseX + offsetX, b.alive && isLastSurvivor, totalTeams);
    }

    // Update overlay
    updateOverlay(frame);
  }

  function updateOverlay(frame) {
    var gapSize = replayData.gap_size || '--';
    var stageId = replayData.stage_id != null ? replayData.stage_id : '--';
    var ep = currentEpisode();
    var totalFrames = ep ? ep.frames.length - 1 : 0;
    var totalEpisodes = replayData.episodes ? replayData.episodes.length : 0;

    if (overlayStage) overlayStage.textContent = stageId;
    if (overlayGap) overlayGap.textContent = gapSize + 'px';
    if (overlayEpisode) overlayEpisode.textContent = (episodeIdx + 1) + ' / ' + totalEpisodes;
    if (overlayFrame) overlayFrame.textContent = frameIdx + ' / ' + totalFrames;

    var alive = 0;
    var birds = frame.birds || [];
    for (var i = 0; i < birds.length; i++) { if (birds[i].alive) alive++; }
    if (overlayAlive) overlayAlive.textContent = alive + ' / ' + birds.length;
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
    ctx.fillText('Run an Admin Race to see replay here', CANVAS_W / 2, CANVAS_H / 2 - 10);
    ctx.font = '400 12px Inter, sans-serif';
    ctx.fillStyle = '#475569';
    ctx.fillText('Upload a model, submit to a stage, then race', CANVAS_W / 2, CANVAS_H / 2 + 14);
  }

  // ── 8. Leaderboard ──────────────────────────────────────────────

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
  loadStagesAndArchitectures();
}
