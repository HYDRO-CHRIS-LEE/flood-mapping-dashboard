# RF Classifier → Flappy Bird Integration Spec (v2)

## 1. Goal

Flappy Bird Competition을 DQN 기반에서 **학생의 RF 분류기 기반**으로 전환한다. 학생이 Classifier 페이지에서 훈련·제출한 RF 모델이 Flappy Bird 에이전트의 두뇌가 되어, 실제 홍수 테스트 샘플을 분류하며 게임을 진행한다.

## 2. Core Concept

```
파이프 하나 = 실제 홍수/비홍수 테스트 샘플 하나

모델이 맞히면 → 새가 갭(홍수 구간)으로 이동 → 통과
모델이 틀리면 → 새가 파이프(비홍수)에 충돌 → Game Over
점수 = 연속 정답 수 = 통과한 파이프 수 (sequential survival)
```

### 교육 메시지

> "너희가 만든 AI 홍수 분류기가 새의 두뇌다. 분류기가 잘 훈련되었다면 새가 매번 홍수 구간(갭)을 정확히 찾아서 통과할 것이고, 그렇지 않으면 비홍수 구간(파이프)에 부딪혀 추락할 것이다."

## 3. Model Lifecycle

### 3.1 모델 적격성 — Option B: 제출된 모델만 사용

Flappy Bird는 **가장 최근에 Classifier 리더보드에 제출(submit)된 모델만** 사용한다. 단순 훈련(train)만으로는 Flappy에 배포할 수 없다. 이유:

- 재현 가능성: 제출 시점의 모델이 기록으로 남음
- 감사 가능성: 리더보드 기록과 게임 성적이 동일 모델에 기반
- 경쟁 공정성: "우연히 좋은 훈련"이 아닌 의도적 제출만 반영

### 3.2 모델 영속화 — 디스크 저장

서버 메모리가 아닌 디스크에 저장한다. 서버 재시작에도 유지.

**저장 경로:** `data/rf_models/{team_id}/model_artifact.pkl`

**Artifact 구조:**

```python
{
    "model": sklearn_rf_object,          # pickled RF classifier
    "features": ["NDWI", "elevation", "slope"],  # 사용된 피처 목록
    "scaler": scaler_or_None,            # StandardScaler/MinMaxScaler or None
    "hyperparameters": {
        "n_trees": 100,
        "max_depth": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": False,
        "scaling": "none",
        "balance": "none",
        "sample_pct": 100,
        "outlier": "none",
    },
    "metrics": {
        "f1": 0.872,
        "accuracy": 0.891,
        "precision": 0.865,
        "recall": 0.880,
    },
    "held_out_events": ["dubai", "germany2021", "libya2023", "china2020"],
    "class_labels": {0: "non-flood", 1: "flood"},
    "artifact_version": "rf_artifact_v1",
    "team_id": "uuid...",
    "team_name": "Team Alpha",
    "created_at": "2026-04-03T14:30:22Z",
}
```

직렬화: `joblib.dump()` / `joblib.load()` (sklearn 모델용 표준).

### 3.3 모델 상태 조회 — 단일 소스

`GET /api/flappy/model-status/{team_id}` 는 디스크의 artifact 파일을 읽어 반환한다. 다른 소스(메모리 캐시, DB)를 참조하지 않는다.

Response:
```json
{
    "ok": true,
    "data": {
        "has_model": true,
        "team_id": "uuid...",
        "team_name": "Team Alpha",
        "features": ["NDWI", "elevation", "slope"],
        "f1": 0.872,
        "accuracy": 0.891,
        "trained_at": "2026-04-03T14:30:22Z",
        "stage_unlocks": [1, 2, 3]
    }
}
```

## 4. Stage System

### 4.1 난이도 — 분류 난이도로 조절

**Confidence 정의:** 각 테스트 샘플에 대해 `confidence = max(predict_proba(sample))`. 이진 RF의 경우 `max(p_flood, p_nonflood)`.

### 4.2 Practice Mode vs Leaderboard Mode

| | Practice Mode | Leaderboard/Race Mode |
|---|---|---|
| 샘플 풀 | 팀 모델의 confidence 기반 (팀별 다름) | **고정 참조 난이도** (전 팀 동일) |
| 리더보드 기록 | 저장 안 함 | 저장함 |
| 목적 | 모델 테스트/연습 | 공정한 경쟁 |

**고정 참조 난이도 (Leaderboard Mode):**

서버 시작 시 또는 최초 레이스 시, 전체 held-out 테스트 샘플을 **기준 RF 모델**(모든 피처 사용, n_trees=200, max_depth=10)로 `predict_proba`를 계산하여 confidence를 매기고, 그 결과를 캐시한다. 이 기준 랭킹이 전 팀 공유 Stage 샘플 풀을 결정한다.

| Stage | 샘플 풀 (기준 confidence 기준) | Pass Avg |
|-------|------|----------|
| 1 | confidence 상위 40% (쉬운 것만) | ≥ 8 |
| 2 | confidence 상위 60% | ≥ 6 |
| 3 | confidence 상위 80% | ≥ 5 |
| 4 | 전체 샘플 | ≥ 4 |
| 5 | confidence 하위 50% (어려운 것만) | ranking only |

### 4.3 Confidence 계산 시점

- 고정 참조 난이도: **서버 시작 시 1회** 계산, 캐시. 데이터 변경 시에만 재계산.
- Practice mode confidence: **모델 배포 시 1회** 계산. 에피소드 큐는 선택된 서브셋 내에서 셔플.

### 4.4 Stage Unlock

Stage N을 통과하려면 **Leaderboard Mode**에서 해당 Stage의 Pass Avg 이상을 달성해야 한다. Stage N+1은 Stage N을 통과한 후에만 잠금 해제.

`GET /api/flappy/stages` 및 `GET /api/flappy/unlocked/{team_id}` 유지.

## 5. Game Engine

### 5.1 에피소드 시뮬레이션

게임 물리는 기존 FlappyBirdEnv 사용:
- gap_size: 150px (전 스테이지 동일)
- speed: dt=0.05 (전 스테이지 동일)
- 물리적 난이도 고정, 분류 품질만으로 승패 결정

각 에피소드:
1. seed로 파이프 위치 결정 (기존 STAGE_SEEDS 사용)
2. Stage의 테스트 샘플 풀에서 seed 기반 셔플
3. 파이프 접근 시 → 큐에서 샘플 추출 → RF predict → 정답/오답 판정
4. 최대 50 파이프/에피소드
5. 10 에피소드/세션

### 5.2 새의 제어 — 결정적 경로

모호한 "최적 flap" 대신, 두 가지 **사전 계산된 결정적 경로**를 사용:

**정답 시 (correct prediction):** `safe_trajectory`
- 현재 새 위치에서 다음 갭 중앙으로 이동하는 flap 시퀀스
- 갭 중앙 ± 10px 이내에 도달하도록 계산
- 통과 후 다음 파이프 대기

**오답 시 (incorrect prediction):** `fail_trajectory`
- flap을 완전히 중단 (중력에 의해 하강)
- 또는 과도한 연속 flap (천장 방향으로 상승)
- 어느 쪽이든 파이프 벽과 충돌 → 게임 종료

두 경로 모두 결정적(deterministic)이므로 리플레이 재생이 정확히 재현 가능.

### 5.3 점수 체계

**Game score = 연속으로 통과한 파이프 수** (sequential survival)

이는 raw classification accuracy와 다름을 명시한다. 추가 메트릭으로 투명성 제공:

| 메트릭 | 설명 |
|--------|------|
| `score` | 연속 통과 파이프 수 (게임 점수) |
| `episode_accuracy` | 해당 에피소드에서 본 샘플 중 정답 비율 |
| `total_correct` | 맞힌 총 샘플 수 |
| `n_samples_seen` | 본 총 샘플 수 (= score + 1, 마지막은 오답) |

## 6. API Design

### 6.1 수정 엔드포인트

**`POST /api/classifier/submit` (수정)**
- 기존: 리더보드에만 기록
- 추가: RF 모델 artifact를 디스크에 저장 (`data/rf_models/{team_id}/model_artifact.pkl`)
- 이를 위해 train 시 모델 객체를 임시 메모리에 보관 → submit 시 디스크에 영속화

**`POST /api/classifier/train` (수정)**
- 기존 동작 유지
- 추가: 훈련 완료 후 모델 객체를 임시 메모리에 보관 (team_id 키), submit 전까지만 유효

### 6.2 신규 엔드포인트

**`GET /api/flappy/model-status/{team_id}`**
- 디스크의 artifact 파일 읽기
- 모델 없으면: `{ok: true, data: {has_model: false}}`

**`POST /api/flappy/play`**
- Request: `{team_id, stage_id, mode: "practice" | "leaderboard"}`
- 실행 후 반환, 리더보드에 저장하지 않음
- Response: scores, replay, metrics

**`POST /api/flappy/save-result`**
- Request: `{team_id, team_name, stage_id, avg_score, max_score, episode_scores, passed}`
- 리더보드에 결과 기록
- play 후 프론트엔드에서 명시적으로 호출

**`POST /api/flappy/race` (수정)**
- DQN 대신 각 팀의 디스크 저장 RF 모델로 실행
- Leaderboard mode 강제
- 결과 자동 저장

### 6.3 유지 엔드포인트

- `GET /api/flappy/stages` — Stage 정보 반환 (gap_size, pass_avg 등)
- `GET /api/flappy/unlocked/{team_id}` — RF 기반 pass 여부로 unlock 계산

### 6.4 삭제 엔드포인트

- `POST /api/flappy/upload` — .pt 파일 업로드
- `POST /api/flappy/submit` (기존 DQN 버전) — 새 save-result로 대체

### 6.5 에러 케이스

| 상황 | HTTP | error code | message |
|------|------|-----------|---------|
| team_id에 모델 없음 | 404 | `MODEL_NOT_FOUND` | "No classifier submitted yet. Train and submit on the Classifier page first." |
| stage_id에 적격 샘플 없음 | 400 | `NO_ELIGIBLE_SAMPLES` | "Not enough test samples available for Stage N." |
| artifact 버전 불일치 | 400 | `ARTIFACT_INCOMPATIBLE` | "Model artifact version mismatch. Please retrain and resubmit." |
| 리플레이 생성 실패 | 500 | `REPLAY_FAILED` | "Failed to generate game replay." |
| 리더보드 저장 실패 | 500 | `LEADERBOARD_SAVE_FAILED` | "Failed to save result to leaderboard." |
| 레이스 진행 중 (lock) | 409 | `RACE_IN_PROGRESS` | "A race is already running. Please wait." |
| 잘못된 admin 비밀번호 | 403 | `INVALID_PASSWORD` | "Invalid admin password." |

## 7. Replay Payload

```json
{
    "stage_id": 3,
    "team_id": "uuid...",
    "team_name": "Team Alpha",
    "mode": "leaderboard",
    "episodes": [
        {
            "episode_index": 0,
            "seed": 3037,
            "final_score": 12,
            "passed": true,
            "episode_accuracy": 0.923,
            "total_correct": 12,
            "n_samples_seen": 13,
            "classification_events": [
                {"pipe_index": 0, "sample_idx": 42, "true_label": 1, "predicted": 1, "correct": true},
                {"pipe_index": 1, "sample_idx": 17, "true_label": 0, "predicted": 0, "correct": true}
            ],
            "frames": [
                {
                    "t": 0,
                    "bird_y": 290.5,
                    "alive": true,
                    "score": 0,
                    "action": 1,
                    "pipes": [{"x": 350.0, "gap_y": 290.0, "gap_size": 150.0}]
                }
            ]
        }
    ],
    "summary": {
        "avg_score": 10.5,
        "max_score": 15,
        "scores": [12, 8, 15, 10, 9, 11, 10, 12, 8, 10]
    }
}
```

## 8. Frontend Changes

### 8.1 삭제

- 아키텍처 선택 UI (model1/model2/model3)
- 하이퍼파라미터 폼 (lr, batch_size, dropout, gamma, episodes)
- .pt/.json 파일 업로드
- DQN Submit to Stage

### 8.2 추가/수정

- **모델 상태 카드**: "Your Classifier: F1 87.2% — NDWI, elevation, slope — submitted 2min ago" 또는 "No classifier submitted yet → Go to Classifier page"
- **Deploy 버튼** + Stage 선택 + Mode 토글 (Practice / Leaderboard)
- **결과 후**: "Save to Leaderboard" 버튼 (leaderboard mode만)
- **에피소드 정확도** 표시: score 외에 accuracy, total_correct도 표시

### 8.3 유지

- Canvas 리플레이 뷰어
- Stage 뱃지 UI
- 리더보드 (5초 폴링)
- Admin Race

## 9. Deletion Targets

**Backend:**
- `backend/services/flappy_engine.py` 내 DQN 관련: `train_demo()`, `validate_upload()`, `submit_model()`
- `backend/routers/flappy.py` 내: `POST /upload`, 기존 `POST /submit`
- `ALLOWED_ARCHITECTURES`, `_model_store` (in-memory dict)

**Frontend:**
- `flappy.js` 내 DQN/upload 관련 전체 코드
- `flappy.html` 내 아키텍처/하이퍼파라미터/업로드 섹션

**유지 (기존 util 모듈):**
- `utils/flappy_eval.py` — STAGES, STAGE_SEEDS, run_race (RF 버전으로 수정)
- `utils/flappy_leaderboard.py` — 리더보드 읽기/쓰기
- `utils/flappy_replay.py` — 리플레이 생성
- `static/flappy_race.html` — Canvas 뷰어 (프론트엔드에 이미 포팅됨)

## 10. Acceptance Criteria

1. Classifier 페이지에서 제출(submit)한 RF 모델만 Flappy Bird에 사용 가능
2. 모델 artifact가 디스크에 영속화되어 서버 재시작 후에도 유지
3. 게임 점수가 sequential classification survival을 반영하며, accuracy 메트릭도 함께 제공
4. Leaderboard mode는 고정 참조 샘플 풀을 사용하여 팀 간 공정성 보장
5. Practice mode는 팀 모델 confidence 기반 난이도 사용
6. 새의 행동이 결정적(deterministic) 경로로 제어되어 리플레이 재현 가능
7. DQN/업로드 관련 코드가 완전히 제거
8. 피처 추가 시 artifact 구조 외에 백엔드 변경 불필요
9. 모든 에러 케이스에 명시된 코드/메시지로 응답
10. Stage unlock이 RF 기반 pass 기준으로 동작
