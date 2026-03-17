# AI Flood Classifier Improvement — Design Spec

**Date**: 2026-03-17
**Scope**: Data leakage 수정 + 고등학생 UX 개선 (AI Classifier 탭)
**Target audience**: 고등학생, 30~45분 수업 세션, 팀 간 경쟁

---

## 1. Problem Statement

### Data Leakage

현재 module4_rf.py는 event-based split으로 spatial leakage를 방지한다고 설명하지만, 피처 자체가 이벤트(지역) 정보를 인코딩하고 있다.

| 피처 | Harvey | Pakistan | Dubai | LA2025 |
|------|--------|----------|-------|--------|
| elevation 평균 | 18m | 535m | 8m | 275m |
| SAR_VH 평균 | -16.5dB | -17.4dB | -19.5dB | -5.1dB |

- elevation만으로 이벤트 식별 ~90% 가능
- 모델이 "홍수 패턴"이 아닌 "지리"를 학습
- LA2025는 water 샘플 3개 — 사실상 사용 불가
- permanent_water는 거의 전부 0 — 상수 피처

### UX 문제

- 결과 숫자만 보여주고 개선 방향 안내 없음
- 리더보드가 accuracy만 추적 — test event 선택으로 점수 부풀리기 가능
- 이전 시도와 비교 불가 — 뭘 바꿨을 때 좋아졌는지 알 수 없음
- 리더보드가 in-memory — 서버 재시작 시 소실

---

## 2. Approach

기능별 모듈 분리 (Approach 2):
- `utils/normalization.py` — z-score 정규화 파이프라인
- `utils/leaderboard.py` — JSON 파일 기반 리더보드
- `module4_rf.py` — ML 로직 + 피드백 힌트 + 히스토리 비교 UI

학생 입장에서는 기존과 동일하게 AI Classifier 탭 하나에서 모든 것을 체험.

---

## 3. Data Leakage 수정 — `utils/normalization.py`

### 이벤트별 z-score 정규화

```python
def normalize_by_event(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    이벤트별 z-score: (값 - 이벤트 평균) / 이벤트 표준편차
    permanent_water는 binary이므로 정규화에서 제외.
    """
```

- `event` 컬럼 기준 groupby
- 표준편차 0인 경우(상수 피처) → 0으로 채움
- `permanent_water` 제외
- 원본 df 유지, 정규화된 복사본을 모델에 전달

### 왜 글로벌 정규화가 아닌가

글로벌 정규화는 선형 변환이라 이벤트 간 클러스터 구조를 보존한다:
```
Harvey: (18 - 209) / 230 ≈ -0.83
Pakistan: (535 - 209) / 230 ≈ +1.42
```
여전히 분리 가능. 이벤트별 정규화만이 절대값 차이를 제거하고 "지역 내 상대 위치" 정보만 보존.

### 최소 샘플 검증

- flood/non-flood 각 30개 미만인 이벤트는 자동 제외
- UI에 회색 처리 + 이유 표시

### 적용 시점

`load_all_rf_samples()` → `normalize_by_event()` → `train_rf()`

---

## 4. 고정 Test Event + 리더보드 — `utils/leaderboard.py`

### Test Event 고정

- `HELD_OUT_EVENT` 상수 (예: `"dubai"`)
- 학생은 test event를 선택할 수 없음 — multiselect 제거
- UI에 고정 표시: "All teams are evaluated on Dubai (2024)"
- train에 사용할 이벤트: 나머지 전부 (최소 샘플 미달 이벤트 제외)

### 리더보드 JSON 구조

파일 위치: `data/leaderboard.json`

```json
{
  "held_out_event": "dubai",
  "entries": [
    {
      "team": "Team Alpha",
      "f1": 0.82,
      "accuracy": 0.85,
      "precision": 0.88,
      "recall": 0.77,
      "features": ["SAR_VH", "NDWI", "elevation"],
      "n_trees": 100,
      "max_depth": 5,
      "timestamp": "2026-03-17T14:30:00"
    }
  ]
}
```

### 정렬 기준

F1 Score (accuracy 대신). Accuracy는 class imbalance에 취약. F1은 precision/recall 균형 요구.

### 팀별 표시

제출 제한 없음. 리더보드에는 팀별 최고 F1만 표시.

### 초기화

수업 전 `data/leaderboard.json` 삭제하면 초기화.

---

## 5. 가이디드 피드백 시스템

### 힌트 규칙

모델 학습 직후, 메트릭 분석하여 최대 2개 힌트 표시.

| 조건 | 힌트 |
|------|------|
| Recall < 0.6 | "Recall이 낮습니다 — 실제 홍수 지역을 많이 놓치고 있어요. SAR_VH 피처가 빠져 있다면 추가해보세요." |
| Precision < 0.6 | "Precision이 낮습니다 — 홍수가 아닌 곳을 홍수로 잘못 예측하고 있어요. elevation이나 slope를 추가하면 도움이 될 수 있어요." |
| F1 > 0.85 | "훌륭합니다! 피처 수를 줄여도 비슷한 성능이 나오는지 실험해보세요." |
| n_trees < 30 and F1 < 0.7 | "트리 수가 적습니다. 50~100으로 늘려보세요." |
| 피처 1개만 사용 | "피처를 하나만 쓰고 있어요. 서로 다른 종류의 정보를 조합해보세요." |

### 구현

`module4_rf.py` 내 `generate_hints(metrics, features)` 함수. 결과 카드 바로 아래 파란색 callout 박스.

---

## 6. 히스토리 비교

### 저장

`st.session_state["run_history"]`에 최대 10개 시도. 각 항목:

```python
{
    "run_id": 1,
    "features": ["SAR_VH", "NDWI"],
    "n_trees": 100,
    "max_depth": 5,
    "f1": 0.78,
    "accuracy": 0.82,
    "precision": 0.85,
    "recall": 0.72,
    "timestamp": "14:32:05"
}
```

### 비교 UI

결과 영역 하단 "Run History" 테이블:

| # | Features | Trees | Depth | F1 | Acc | vs Prev |
|---|----------|-------|-------|----|-----|---------|
| 3 | SAR, NDWI, elev | 100 | 5 | 82.1% | 85.3% | **+4.2%** |
| 2 | SAR, NDWI | 50 | 5 | 77.9% | 81.0% | -1.5% |
| 1 | SAR, NDWI | 100 | 5 | 79.4% | 82.2% | — |

- vs Prev: 직전 시도 대비 F1 변화 (초록/빨강)
- 최신 시도 상단
- 피처 변경 시 해당 셀 볼드 처리

### 영속성

세션 내 `st.session_state`만 사용. JSON 저장 불필요.

---

## 7. 파일 변경 요약

### 새 파일

| 파일 | 역할 | 예상 크기 |
|------|------|----------|
| `utils/normalization.py` | 이벤트별 z-score 정규화, 최소 샘플 검증 | ~50줄 |
| `utils/leaderboard.py` | JSON 파일 기반 리더보드 CRUD | ~60줄 |

### 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `modules/module4_rf.py` | test event 고정, 정규화 적용, 힌트 시스템, 히스토리 비교 UI, 리더보드 F1 기준 + JSON 연동 |
| `utils/data_loader.py` | `load_all_rf_samples()`에 최소 샘플 검증 로직 추가 |

### 변경하지 않는 파일

- `app.py` — navigation 구조 그대로
- `module1_sar.py`, `module2_optical.py`, `module6_gpm.py` — 변경 없음
- `utils/styles.py` — 기존 CSS 재활용

---

## 8. 데이터 흐름

```
load_all_rf_samples(available_events)
  → 최소 샘플 검증 (flood/non-flood 각 30개 미만 이벤트 제외 + 경고)
  → normalize_by_event(df, features)
  → train_rf(df_norm, features, HELD_OUT_EVENT)
  → metrics + importance
  → generate_hints(metrics, features)
  → st.session_state["run_history"]에 저장
  → leaderboard.json에 제출 (선택)
```

---

## 9. 데이터 의존성

이 spec은 "데이터가 N개 이상 이벤트일 때" 동작하는 구조를 설계한다. 추가 홍수 이벤트 데이터 확보(GEE Colab 노트북 실행)는 별도 태스크로 분리.

현재 사용 가능: harvey (710 samples), pakistan (800), dubai (735), la2025 (330, water 3개 — 자동 제외 대상)
