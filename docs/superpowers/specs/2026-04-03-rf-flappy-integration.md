# RF Classifier → Flappy Bird Integration Spec

## 1. Goal

Flappy Bird Competition을 DQN 기반에서 **학생의 RF 분류기 기반**으로 전환한다. 학생이 Classifier 페이지에서 훈련한 RF 모델이 Flappy Bird 에이전트의 두뇌가 되어, 실제 홍수 테스트 샘플을 분류하며 게임을 진행한다.

## 2. Core Concept

```
파이프 하나 = 실제 홍수/비홍수 테스트 샘플 하나

모델이 맞히면 → 새가 갭(홍수 구간)으로 이동 → 통과
모델이 틀리면 → 새가 파이프(비홍수)에 충돌 → Game Over
점수 = 연속으로 맞힌 샘플 수 = 통과한 파이프 수
```

### 교육 메시지

> "너희가 만든 AI 홍수 분류기가 새의 두뇌다. 분류기가 잘 훈련되었다면 새가 매번 홍수 구간(갭)을 정확히 찾아서 통과할 것이고, 그렇지 않으면 비홍수 구간(파이프)에 부딪혀 추락할 것이다."

## 3. Game Flow

### 3.1 사전 조건

학생이 Classifier 페이지에서 RF 모델을 훈련하고 결과를 확인한 상태여야 한다. 훈련 시 서버는 RF 모델 객체 + 사용된 피처 목록 + scaler를 team_id 키로 메모리에 저장한다.

### 3.2 게임 실행 흐름

1. 학생이 Flappy Bird 페이지에서 "Deploy My Classifier" 클릭
2. 서버가 team_id로 저장된 RF 모델 로드
3. 서버가 held-out 테스트 샘플을 Stage 난이도에 맞게 정렬 (아래 3.3 참조)
4. 게임 시뮬레이션 실행:
   - 파이프에 접근할 때마다 큐에서 샘플 하나를 꺼냄
   - RF 모델이 해당 샘플을 분류 (predict)
   - **정답이면**: 새가 갭 중앙으로 이동 (통과)
   - **오답이면**: 새가 파이프 벽으로 이동 (충돌, 게임 종료)
5. 10 에피소드 실행 (각 에피소드마다 샘플 큐를 셔플, seed 고정)
6. 결과 반환: 에피소드별 점수, 평균 점수, 리플레이 데이터

### 3.3 Stage 난이도 — 분류 난이도로 조절

모든 테스트 샘플에 대해 `RF.predict_proba()`를 실행하여 confidence를 측정하고, confidence 기준으로 난이도 티어를 나눈다:

| Stage | 샘플 선택 기준 | 설명 |
|-------|--------------|------|
| 1 | confidence 상위 40% (쉬운 것만) | 명확한 flood/non-flood |
| 2 | confidence 상위 60% | 쉬운 + 보통 |
| 3 | confidence 상위 80% | 경계 케이스 일부 포함 |
| 4 | 전체 샘플 | 어려운 것 포함 |
| 5 | confidence 하위 50% (어려운 것만) | edge case 집중 |

**Pass 기준** (평균 점수):

| Stage | Pass Avg |
|-------|----------|
| 1 | ≥ 8 |
| 2 | ≥ 6 |
| 3 | ≥ 5 |
| 4 | ≥ 4 |
| 5 | ranking only |

Pass 기준이 Stage가 올라갈수록 낮아지는 이유: 샘플 난이도가 올라가므로 같은 모델이라도 더 적게 맞힌다.

### 3.4 에피소드 시뮬레이션

게임 물리는 기존과 동일 (FlappyBirdEnv 사용):
- gap_size: 전 스테이지 동일 (150px)
- speed: 전 스테이지 동일 (dt=0.05)
- 파이프 간격: 동일

물리적 난이도는 고정하고, **분류 품질만으로 승패가 결정**된다.

각 에피소드:
1. seed로 파이프 위치 결정 (기존 STAGE_SEEDS 사용)
2. 테스트 샘플 큐를 seed로 셔플
3. 파이프 접근 시 → 샘플 분류 → 맞으면 최적 경로, 틀리면 충돌 경로
4. 최대 파이프 수: 50개 (에피소드당)
5. 프레임 기록 (리플레이용)

### 3.5 새의 물리적 행동

**정답 시 (올바르게 분류):**
- 다음 N프레임 동안 갭 중앙을 향해 최적 flap 시퀀스 실행
- 실질적으로 "자동 조종"으로 갭 통과

**오답 시 (잘못 분류):**
- flap을 하지 않거나 반대 방향으로 이동
- 중력에 의해 파이프에 충돌 → 게임 종료

## 4. Backend Changes

### 4.1 모델 저장 (`POST /api/classifier/train` 수정)

훈련 완료 후 반환하기 전에 서버 메모리에 저장:

```python
# Module-level store
_model_store: dict[str, dict] = {}
# key: team_id
# value: {"model": sklearn RF object, "features": [...], "scaler": scaler_or_None}
```

### 4.2 신규 엔드포인트

**`POST /api/flappy/play`**
- Request: `{team_id, stage_id}`
- 저장된 RF 모델 로드
- 테스트 샘플 로드 + confidence 기반 필터링
- 10 에피소드 실행
- Response: `{ok, data: {scores: [...], avg_score, max_score, passed, replay}}`

**`POST /api/flappy/race`** (수정)
- 모든 팀의 저장된 RF 모델로 동시 레이스
- DQN 코드 제거

**`GET /api/flappy/model-status/{team_id}`** (신규)
- 모델이 저장되어 있는지, 어떤 피처로 훈련했는지, F1 점수 확인
- Response: `{ok, data: {has_model, features, f1, accuracy}}`

### 4.3 새 서비스: `backend/services/rf_game_engine.py`

기존 `flappy_engine.py`의 DQN 부분을 RF 기반으로 교체:

```python
def run_rf_episode(model, scaler, features, test_samples, env, seed, gap_size, dt, max_pipes=50, record_frames=True) -> dict

def select_samples_for_stage(model, scaler, features, all_samples, stage_id) -> list

def run_rf_game(team_model_data, stage_id, data_root, n_episodes=10) -> dict
```

### 4.4 삭제 대상

- `train_demo()` 함수 (DQN 훈련)
- `validate_upload()`, `submit_model()` (파일 업로드)
- `POST /api/flappy/upload` 엔드포인트
- `POST /api/flappy/submit` 엔드포인트
- `ALLOWED_ARCHITECTURES` 관련 코드

## 5. Frontend Changes

### 5.1 `flappy.html` 수정

제거:
- 아키텍처 선택 (model1/model2/model3)
- 하이퍼파라미터 폼 (lr, batch_size, dropout, gamma, episodes)
- 파일 업로드 (.pt, .json)
- Submit to Stage 버튼

추가:
- **모델 상태 표시**: "Your Classifier: F1 87.2% — NDWI, elevation, slope" 또는 "No classifier trained yet — go to AI Flood Classifier first"
- **"Deploy My Classifier" 버튼** + Stage 선택
- 결과: 에피소드별 점수 + 리플레이

### 5.2 `flappy.js` 수정

- DQN/업로드 로직 전부 제거
- `init_flappy()`:
  1. `GET /api/flappy/model-status/{team_id}` 로 모델 상태 확인
  2. 모델 있으면 Deploy 버튼 활성화, 없으면 Classifier 페이지 안내
  3. Deploy 클릭 → `POST /api/flappy/play` → 리플레이 렌더링
  4. 리더보드 폴링 유지

### 5.3 유지

- Canvas 리플레이 뷰어 (그대로)
- Stage 뱃지 UI (그대로)
- 리더보드 UI (그대로)
- Admin Race 기능 (RF 기반으로 전환)

## 6. 리플레이 데이터 형식

기존 형식과 동일. 새의 y 위치, alive 상태, 파이프 위치 등은 게임 엔진이 프레임별로 기록. RF의 분류 결과에 따라 새의 행동만 달라질 뿐, 리플레이 데이터 구조는 변경 없음.

추가 필드 (optional, 시각화용):
```json
{
  "t": 0,
  "bird_y": 290.5,
  "alive": true,
  "score": 3,
  "action": 1,
  "classification": {
    "sample_idx": 42,
    "true_label": 1,
    "predicted": 1,
    "correct": true
  }
}
```

## 7. Acceptance Criteria

1. Classifier 페이지에서 훈련한 RF 모델이 Flappy Bird 에이전트로 사용된다
2. 게임 점수가 실제 홍수 분류 정확도를 직접 반영한다
3. Stage 난이도가 분류 난이도(confidence 기반)로 조절된다
4. DQN/업로드 관련 코드가 완전히 제거된다
5. 기존 Canvas 리플레이 뷰어가 정상 동작한다
6. 리더보드가 RF 기반 점수로 업데이트된다
7. 피처 추가 시 백엔드 변경 없이 동작한다 (학생이 새 피처를 선택하면 모델이 자동으로 사용)
