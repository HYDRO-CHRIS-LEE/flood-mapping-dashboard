# EarthAI: Streamlit → FastAPI + Vanilla Frontend Migration (Design Spec v2)

## 1. Goal

EarthAI flood dashboard를 Streamlit에서 **FastAPI backend + vanilla HTML/CSS/JS frontend** 구조로 이관한다.
목표는 다음 3가지다:

1. stitch HTML mockup을 최대한 그대로 구현할 수 있는 **완전한 UI 제어권 확보**
2. 기존 Python 데이터 처리 및 모델 로직의 **최대 재사용**
3. 약 **20명 규모 교실 환경**에서 안정적으로 동작하는 수업용 웹앱 구축

---

## 2. System Architecture

```text
Browser (HTML/CSS/JS)         FastAPI (Python)
----------------------        ----------------------------
Vanilla SPA shell             /api/events
Stitch-based pages            /api/rainfall/{event}
Leaflet / Plotly              /api/tif/{event}/{layer}/{period}
sessionStorage + team_id      /api/sar/*
Hash routing                  /api/classifier/*
                              /api/flappy/*
                              /api/leaderboard/*
                              SQLite
                              In-memory / file cache
```

**Design principle:**
Frontend는 UI 렌더링과 사용자 상호작용만 담당하고, backend는 데이터 제공·계산·저장을 담당한다.
기존 Streamlit 로직은 가능한 한 `services/` 아래의 **pure Python functions**로 분리하여 재사용한다.

---

## 3. Frontend Design

### 3.1 SPA shell

* `index.html` 하나를 공통 shell로 사용
* 공통 요소:

  * 좌측 sidebar
  * 상단 topbar
  * profile/team controls
  * `<main id="page-content">`

### 3.2 Routing

* hash routing 사용: `#rainfall`, `#optical`, `#sar`, `#classifier`, `#flappy`
* route 변경 시 page HTML을 로드하고, 해당 JS `init()` 실행

### 3.3 State

* `sessionStorage`에는 **display name**과 UI 상태만 저장
* 내부 식별은 별도의 `team_id` UUID를 사용
* leaderboard 제출, flappy unlock, session tracking은 `team_id` 기준으로 처리

### 3.4 Assets

* 개발 단계에서는 CDN 사용 가능
* 배포판에서는 주요 라이브러리(Tailwind, Alpine, Leaflet, Plotly)는 **local fallback 또는 vendored assets** 제공
* 목표: 교실 네트워크 환경에서도 최소 기능 보장

---

## 4. Backend Design

### 4.1 App structure

* `main.py`: FastAPI app, middleware, static mount, router registration
* `routers/`: API endpoint 정의
* `services/`: 기존 Streamlit 로직 이관
* `db.py`: SQLite init / migration
* `config.py`: paths, constants, feature flags

### 4.2 CORS / serving

* 개발 환경: permissive CORS 허용
* 운영 환경: same-origin 또는 지정 origin만 허용
* FastAPI가 static frontend와 API를 함께 서빙

### 4.3 Response standard

모든 JSON 응답은 아래 형식을 따른다:

```json
{ "ok": true, "data": {...} }
```

오류 시:

```json
{ "ok": false, "error": "EVENT_NOT_FOUND", "message": "..." }
```

---

## 5. Performance and Concurrency Policy

### 5.1 Request classes

* **Fast**: events, rainfall, bounds, leaderboard 조회
* **Medium**: SAR compute, classifier train
* **Heavy**: flappy train, flappy race

### 5.2 Execution policy

* Fast/Medium 요청은 일반 API 요청으로 처리
* Heavy 요청은 **single-flight 제한** 또는 **job-based execution** 적용
* 한 팀이 동시에 여러 flappy training을 실행하지 못하도록 제한

### 5.3 Caching

다음은 캐시 대상이다:

* event metadata
* rainfall timeseries
* optical bounds
* rendered PNG tiles
* SAR compute 결과 (`event + threshold + remove_permanent`)
* leaderboard 조회 결과(짧은 TTL)

기존 `@st.cache_data` 제거 후 성능을 유지하려면, 이 캐시 계층이 필요하다.

---

## 6. API Summary

### Core

* `GET /api/events`
* `GET /api/events/{key}`

### Rainfall

* `GET /api/rainfall/{event}`

### Optical

* `GET /api/tif/{event}/{layer}/{period}`
* `GET /api/tif/{event}/bounds`

**Note:** bounds는 JSON endpoint로만 제공하고, PNG endpoint는 이미지 바이트만 반환한다.

### SAR

* `POST /api/sar/compute`
* `GET /api/sar/tile/{event}/{threshold}/{remove_perm}`

### Classifier

* `POST /api/classifier/train`
* `POST /api/classifier/submit`

### Flappy

* `POST /api/flappy/train`
* `POST /api/flappy/upload`
* `POST /api/flappy/submit`
* `POST /api/flappy/race`
* `GET /api/flappy/stages`
* `GET /api/flappy/unlocked/{team_id}`

### Leaderboard

* `GET /api/leaderboard/classifier`
* `GET /api/leaderboard/flappy/{stage_id}`

---

## 7. Page Scope

### Rainfall

* Plotly rainfall chart
* flood window highlight
* threshold-based recoloring
* lightweight, client-side interaction 중심

### Optical

* Leaflet image overlay
* RGB / NDWI / NDWI change toggle
* before / after toggle
* bounds JSON + PNG overlay 조합 사용

### SAR

* threshold slider
* Otsu auto-threshold
* permanent water removal toggle
* flood metrics + overlay 갱신

### Classifier

* feature selection + hyperparameter form
* synchronous RF training
* confusion matrix, metrics, feature importance 표시
* **"SHAP" 표기는 제거하고 "Feature Importance"로 통일**

### Flappy

* training / upload / submit / replay
* 가장 무거운 페이지이므로 마지막 단계 구현
* 교사용 admin 기능은 최소 범위로 유지

---

## 8. Data and Storage

### SQLite

* classifier leaderboard
* flappy leaderboard
* optional migration from legacy JSON

### Temp artifacts

* flappy uploaded models
* race replay JSON
* generated tiles (optional cache directory)

### Cleanup policy

* temp model / replay / cache 파일은 TTL 기반 정리
* 서버 재시작 시 복구 불필요한 임시 산출물은 삭제 가능

---

## 9. Implementation Phasing

### Phase 1 — Core Infrastructure

* FastAPI scaffold
* SPA shell
* router/app.js/api.js
* events API
* SQLite init
* 공통 error/response 규약 적용

### Phase 2 — Rainfall + Optical

* data loader 이관
* rainfall page
* optical page
* raster/bounds cache 추가

### Phase 3 — SAR + Classifier

* SAR engine 이관
* RF engine 이관
* leaderboard 연결
* medium endpoint 성능 점검

### Phase 4 — Flappy

* flappy engine 이관
* upload/submit/race
* heavy job 제한 정책 적용
* classroom stress test 수행

---

## 10. Acceptance Criteria

이관 완료 기준은 아래와 같다:

1. stitch 기반 레이아웃이 Streamlit 대비 충분히 충실하게 재현된다.
2. 기존 Python 로직이 FastAPI `services/`에서 재사용된다.
3. 최소 20명 규모 수업 환경에서 기본 탐색·지도·차트·leaderboard가 안정적으로 동작한다.
4. flappy / classifier / SAR 요청이 전체 앱 응답성을 무너뜨리지 않는다.
5. 오류 시 사용자에게 일관된 메시지가 표시된다.
6. 운영 배포에서 CDN 문제 없이 핵심 기능이 유지된다.

---

## Project Structure

```
EarthAI/
├── backend/
│   ├── main.py              # FastAPI app, CORS, static mount, router registration
│   ├── config.py            # DATA_ROOT, DB path, ALL_EVENTS, constants, feature flags
│   ├── db.py                # SQLite init, connection helper, migration from legacy JSON
│   ├── routers/
│   │   ├── events.py        # GET /api/events, GET /api/events/{key}
│   │   ├── rainfall.py      # GET /api/rainfall/{event}
│   │   ├── optical.py       # GET /api/tif/{event}/{layer}/{period}, bounds
│   │   ├── sar.py           # POST /api/sar/compute, GET /api/sar/tile/...
│   │   ├── classifier.py    # POST /api/classifier/train, /submit
│   │   ├── flappy.py        # POST /api/flappy/train, /upload, /submit, /race, GET stages/unlocked
│   │   └── leaderboard.py   # GET /api/leaderboard/classifier, /flappy/{stage_id}
│   └── services/
│       ├── data_loader.py   # From utils/data_loader.py — GeoTIFF/CSV, no @st.cache_data
│       ├── normalization.py # From utils/normalization.py — z-score normalization
│       ├── sar_engine.py    # From module1 — threshold, Otsu, flood mask → numpy/PNG
│       ├── optical_engine.py# From module2 — tile generation → PNG bytes
│       ├── rf_engine.py     # From module4 — RF training, metrics, importance
│       ├── flappy_engine.py # From module5 — DQN training, race runner, replay
│       └── cache.py         # In-memory/file caching layer (replaces @st.cache_data)
├── frontend/
│   ├── index.html           # SPA shell: sidebar + topbar + <main id="page-content">
│   ├── pages/
│   │   ├── rainfall.html
│   │   ├── optical.html
│   │   ├── sar.html
│   │   ├── classifier.html
│   │   └── flappy.html
│   ├── css/
│   │   └── style.css        # Consolidated from stitch designs + Tailwind overrides
│   └── js/
│       ├── app.js           # SPA router, page loader, sidebar state, profile dropdown
│       ├── api.js           # fetch wrapper: base URL, X-Team-Id header, error handling
│       ├── rainfall.js      # init(): fetch data → Plotly chart + controls
│       ├── optical.js       # init(): Leaflet map + layer/period toggle
│       ├── sar.js           # init(): Leaflet map + threshold slider → API → update
│       ├── classifier.js    # init(): form → train API → render results
│       └── flappy.js        # init(): form, upload, submit, replay viewer
├── data/                    # Existing GeoTIFF/CSV data (unchanged)
│   └── earthai.db           # SQLite (created on first run)
├── requirements.txt
└── run.py                   # Entry point: uvicorn backend.main:app
```

## SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS classifier_leaderboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id TEXT NOT NULL,
    team_name TEXT NOT NULL,
    f1 REAL NOT NULL,
    accuracy REAL,
    precision_val REAL,
    recall REAL,
    features TEXT,
    n_trees INTEGER,
    max_depth INTEGER,
    test_events TEXT,
    submitted_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS flappy_leaderboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id TEXT NOT NULL,
    team_name TEXT NOT NULL,
    stage_id INTEGER NOT NULL,
    avg_score REAL NOT NULL,
    max_score INTEGER,
    survival_steps_avg REAL,
    episode_scores TEXT,
    passed INTEGER DEFAULT 0,
    race_id TEXT,
    submitted_at TEXT DEFAULT (datetime('now'))
);
```
