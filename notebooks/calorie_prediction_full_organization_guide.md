# 🔥 칼로리 예측 프로젝트 — 전체 파일 정리 가이드 (최종)

## 전체 실험 흐름 (6단계)

```
Phase 1: 트리 기반 모델 (RMSE ~1.37)
  → DT/RF/XGB/LGBM 비교 → LGBM 채택 → log변환 → 파생변수 → Optuna → Seed Ensemble

Phase 2: EDA 재수행
  → 5단계 체계적 분석 → 타겟 우측 편향 확인 → 변수 관계 재정리

Phase 3: Polynomial Regression 도입 (RMSE 0.92 → 0.31)
  → Poly degree 비교 → Ridge 규제 → KFold → 타겟 변환 다양화 스태킹

Phase 4: Feature Pruning + 모델 경량화 (RMSE ~0.30)
  → TOP5 파생변수 선별 → base+TOP5 pruning → identity/sqrt 결합

Phase 5: 구조 분석 + 성별 분리 (RMSE ~0.17)
  → 잔차 분석으로 곱셈 구조 발견 → 핵심 5변수만 사용 → 성별 분리 모델
  → LightGBM+Keyser 공식, MLP 시도 (성능 개선 미미)

Phase 6: 최종 최적화 (RMSE 0.13, LB 0.12)
  → 성별별 degree/alpha 그리드 탐색 → round RMSE 최적화 → 정수 타겟 반영
```

---

## 📂 전체 원본 파일 → 통합 매핑

### 🟢 notebooks/01_EDA.ipynb

| 원본 파일 | 비고 |
|-----------|------|
| `calories_burned_eda_ver2.ipynb` | **메인 EDA** |
| `calories_burned_eda_ver2_복사본.ipynb` | 중복 삭제 |

**요약:** 결측치/중복 점검, 타겟 우측 편향(skew 0.82), Height 통합, 상관관계 분석, 이상치 탐지, 정규성 검정

---

### 🟡 notebooks/02_Baseline_Tree_Model.ipynb

| 원본 파일 | 핵심 내용 |
|-----------|----------|
| `basemodel_ver1.ipynb` | 4모델 비교 + 상관계수 분석 |
| `_Baseline__DecisionTree_Regressor_황진영.ipynb` | DT/RF/XGB/LGBM + 로그변환 + 파생변수 |
| `_Baseline__DecisionTree_Regressor_파생변수추가본_황진영.ipynb` | 파생변수 추가 실험 |
| `_Baseline__DecisionTree_Regressor_Seed_5_Ensemble_황진영.ipynb` | Seed Ensemble |
| `calories_burned_model_ver2.ipynb` | XGBoost+Optuna+Seed Ensemble |
| `calories_burned_model_ver3.ipynb` | LGBM+XGB 앙상블 |

**요약:** LGBM 최우수(RMSE 2.24) → log변환(1.99) → 파생변수(1.78) → Optuna(~1.37) → 트리 한계 확인

---

### 🔵 notebooks/03_Polynomial_Experiment.ipynb

| 원본 파일 | 핵심 내용 | RMSE |
|-----------|----------|------|
| `calories_burned_ver4_그만하고싶다.ipynb` | Poly 첫 시도 | - |
| `Calories_Burned_ver5.ipynb` | Poly 파이프라인 | - |
| `calories_burned_ver5_황진영.ipynb` | 위와 동일 | - |
| `calories_burned_ver6_1.ipynb` | weight 파생+degree 2 | 0.537 |
| `calories_burned_ver6_2.ipynb` | ver6_1 중복 | - |
| `calories_burned_ver6_3.ipynb` | degree 3+KFold | 0.581 |
| `calories_burned_ver6_4_alpha.ipynb` | 이상치 처리 실험 | - |
| `calories_burned_ver6_kfold.ipynb` | KFold alpha 튜닝 | - |
| `calories_burned_ver6_5.ipynb` | Degree 앙상블+스태킹 시작 | - |
| `calories_burned_ver6_6_수정.ipynb` | 타겟 변환 다양화 스태킹 | 0.318 |
| `calories_burned_ver6_7.ipynb` | ver6_6 중복 | - |
| `calories_burned_ver6_8.ipynb` | Stratified KFold 실험 | - |
| `calories_burned_ver6_8_cleaned.ipynb` | Log-space Meta 실험 | 0.318 |
| `calories_burned_ver6_8_cleaned_복사본.ipynb` | 가중치최적화+Optuna+FeatureSelection | 0.309 |
| `calories_burned_ver6_9.ipynb` | 가중치 최적화+OneHot개선 | 0.309 |
| `calories_burned_ver6_9_1.ipynb` | TOP5 pruning + identity/sqrt 결합 | 0.299 |
| `calories_burned_ver6_9_cleaned.ipynb` | 정리본+상호작용 시각화 | 0.300 |

**요약:** Poly(deg 1/2/3) 비교 → Ridge 규제 → KFold → Degree 앙상블 → 타겟 변환 스태킹(0.318) → 가중치 최적화(0.309) → Pruning(0.299)

---

### 🟣 notebooks/04_Model_Refinement.ipynb

| 원본 파일 | 핵심 내용 | RMSE |
|-----------|----------|------|
| `calories_burned_ver7_황진영.ipynb` | base+TOP5 pruning + identity/sqrt Ridge 결합 | 0.299 |
| `calories_burned_ver7_1.ipynb` | 위와 동일 구조 + round 적용 | 0.221 |
| `calories_burned_ver7_d4_mlp.ipynb` | Degree 4 Ridge + MLP 앙상블 시도 | - |
| `calories_burned_ver7_lgb_keyser.ipynb` | LightGBM+Keyser 공식+Ridge 앙상블 | - |
| `calories_burned_ver8.ipynb` | 잔차 구조 분석 (곱셈 구조 발견) + round 보정 | 0.218 |
| `calories_burned_ver8_1_015.ipynb` | 성별 분리 + round + tie-break 보정 | 0.157 |
| `calories_burned_ver8_1_017.ipynb` | ver8_1_015와 동일 구조 | 0.157 |

**요약:** 파생변수 pruning → identity/sqrt 결합 → round 적용(0.221) → 잔차 분석으로 곱셈 구조 발견 → 성별 분리 첫 시도(0.157)

---

### 🔴 notebooks/05_Final_Model.ipynb

| 원본 파일 | 핵심 내용 | OOF RMSE | LB |
|-----------|----------|----------|-----|
| `calories_burned_ver8_2.ipynb` | 성별별 degree/alpha 그리드 + round RMSE 최적화 | 0.134 | 0.12 |
| `calories_burned_ver8_3.ipynb` | ver8_2 + 상세 주석 + Final Summary | **0.134** | **0.12** |

**요약:** 성별 분리(M/F) + 핵심 5변수(Exercise_Duration, BPM, Temp_diff, Age, Weight) + Poly(deg 2)+Ridge + 정수 반올림 → **최종 RMSE 0.134, LB 0.12**

---

## 📁 최종 깃허브 구조

```
calorie-prediction/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                              # train.csv, test.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb                      # 탐색적 데이터 분석
│   ├── 02_Baseline_Tree_Model.ipynb      # DT/RF/XGB/LGBM 비교
│   ├── 03_Polynomial_Experiment.ipynb    # Poly + Ridge + 스태킹
│   ├── 04_Model_Refinement.ipynb         # Pruning + 잔차 분석 + 성별 분리
│   └── 05_Final_Model.ipynb              # 최종 모델 (RMSE 0.13, LB 0.12)
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── inference.py
│
├── results/
│   ├── model_comparison.csv
│   └── plots/
│
└── archive/                              # 실험 기록 (.gitignore 또는 별도 branch)
    ├── 01_baseline_4model_comparison.ipynb
    ├── 02_baseline_feature_engineering.ipynb
    ├── 03_baseline_seed5_ensemble.ipynb
    ├── 04_xgboost_optuna_tuning.ipynb
    ├── 05_lgbm_xgb_ensemble.ipynb
    ├── 06_polynomial_first_attempt.ipynb
    ├── 07_poly_degree2_weight_features.ipynb
    ├── 08_poly_degree3_kfold_ridge.ipynb
    ├── 09_poly_outlier_experiment.ipynb
    ├── 10_poly_kfold_alpha_tuning.ipynb
    ├── 11_target_transform_stacking.ipynb
    ├── 12_weight_optimization_scipy.ipynb
    ├── 13_feature_pruning_top5.ipynb
    ├── 14_identity_sqrt_ridge_blend.ipynb
    ├── 15_residual_analysis_multiply_structure.ipynb
    ├── 16_degree4_mlp_experiment.ipynb
    ├── 17_lightgbm_keyser_formula.ipynb
    ├── 18_gender_split_first_attempt.ipynb
    └── 19_gender_split_grid_search_final.ipynb
```

---

## 🗂 archive/ 상세 매핑

| # | archive 파일명 | 원본 | 핵심 내용 |
|---|---------------|------|----------|
| 01 | baseline_4model_comparison | basemodel_ver1 / _Baseline__DT_황진영 | DT/RF/XGB/LGBM 비교 (RMSE 2.24) |
| 02 | baseline_feature_engineering | _Baseline__DT_파생변수추가본 | Intensity, Effort 파생변수 (RMSE 1.78) |
| 03 | baseline_seed5_ensemble | _Baseline__DT_Seed_5_Ensemble | 5개 시드 앙상블 |
| 04 | xgboost_optuna_tuning | model_ver2 | XGBoost+Optuna+LGBM 재실험 |
| 05 | lgbm_xgb_ensemble | model_ver3 | LGBM+XGB 앙상블 |
| 06 | polynomial_first_attempt | ver4/ver5/ver5_황진영 | Poly Regression 첫 시도 |
| 07 | poly_degree2_weight_features | ver6_1/ver6_2 | weight 파생변수+degree 2 |
| 08 | poly_degree3_kfold_ridge | ver6_3 | degree 3+KFold (RMSE 0.58) |
| 09 | poly_outlier_experiment | ver6_4_alpha | 이상치 처리 실험 |
| 10 | poly_kfold_alpha_tuning | ver6_kfold | KFold+alpha logspace 튜닝 |
| 11 | target_transform_stacking | ver6_5/ver6_6_수정/ver6_7 | 타겟변환 다양화 스태킹 (RMSE 0.318) |
| 12 | weight_optimization_scipy | ver6_8*/ver6_9 | scipy 가중치 최적화 (RMSE 0.309) |
| 13 | feature_pruning_top5 | ver6_9_1/ver6_9_cleaned | TOP5 pruning+시각화 (RMSE 0.300) |
| 14 | identity_sqrt_ridge_blend | ver7_황진영/ver7_1 | identity/sqrt Ridge 결합 (RMSE 0.299→0.221) |
| 15 | residual_analysis_multiply_structure | ver8 | 잔차 분석→곱셈 구조 발견 |
| 16 | degree4_mlp_experiment | ver7_d4_mlp | Degree 4+MLP 앙상블 시도 |
| 17 | lightgbm_keyser_formula | ver7_lgb_keyser | LGBM+Keyser 공식+Ridge 앙상블 |
| 18 | gender_split_first_attempt | ver8_1_015/ver8_1_017 | 성별 분리+round 보정 (RMSE 0.157) |
| 19 | gender_split_grid_search_final | ver8_2/ver8_3 | 성별별 그리드탐색 (RMSE 0.134, LB 0.12) |

---

## ⚠️ 삭제 가능한 완전 중복 파일

| 삭제 대상 | 원본과 동일 |
|-----------|------------|
| `calories_burned_eda_ver2_복사본.ipynb` | eda_ver2와 동일 |
| `Calories_Burned_ver5.ipynb` | ver5_황진영과 동일 |
| `calories_burned_ver6_2.ipynb` | ver6_1과 동일 |
| `calories_burned_ver6_7.ipynb` | ver6_6_수정과 동일 |
| `calories_burned_ver6_8_cleaned_복사본.ipynb` | ver6_8_cleaned의 복사본 |
| `calories_burned_ver8_1_017.ipynb` | ver8_1_015와 동일 |

---

## 📊 성능 추이 (RMSE)

| 단계 | 모델 | OOF RMSE | LB | 핵심 개선 |
|------|------|----------|-----|----------|
| 1 | LGBM (기본) | 2.241 | - | 베이스라인 |
| 2 | LGBM + log + 파생변수 | 1.775 | - | 로그변환+Intensity |
| 3 | LGBM + Optuna | ~1.37 | - | 하이퍼파라미터 튜닝 |
| 4 | Poly(deg3) + Ridge | 0.920 | - | 모델 전환 (핵심!) |
| 5 | Poly + Ridge + KFold | 0.537 | - | 교차검증 안정화 |
| 6 | 타겟 변환 스태킹 | 0.318 | - | log/sqrt/YJ/raw 결합 |
| 7 | 가중치 최적화 | 0.309 | 0.31 | scipy optimize |
| 8 | Feature pruning (TOP5) | 0.300 | 0.30 | 노이즈 제거 |
| 9 | identity/sqrt 결합+round | 0.221 | 0.22 | 정수 타겟 반영 |
| 10 | 성별 분리 첫 시도 | 0.157 | 0.20 | M/F 분리 모델 |
| 11 | **성별별 그리드 탐색** | **0.134** | **0.12** | degree/alpha 최적화 |

---

## 💡 notebooks/ 각 파일 헤더 요약 (README용)

**01_EDA.ipynb** — 탐색적 데이터 분석
- 데이터 품질: 결측치 0, 중복 0
- 타겟 우측 편향(skew 0.82) → 로그 변환 필요
- 핵심 변수: Exercise_Duration, Body_Temp, BPM
- Height 통합, 이상치 점검, 정규성 검정

**02_Baseline_Tree_Model.ipynb** — 트리 모델 비교
- DT/RF/XGBoost/LightGBM 4종 비교
- LGBM 최우수 → Optuna 튜닝 → Seed Ensemble
- 트리 모델 한계 확인 (RMSE ~1.37)

**03_Polynomial_Experiment.ipynb** — 다항 회귀 실험
- 모델 전환: 트리 → Polynomial Regression
- Poly degree 1/2/3 비교 → Ridge 규제 → KFold
- 타겟 변환 다양화 스태킹 → scipy 가중치 최적화
- Feature pruning (base + TOP5) → RMSE 0.30

**04_Model_Refinement.ipynb** — 모델 정교화
- 잔차 분석: 변수 간 곱셈 구조 발견
- identity/sqrt Ridge 결합 + round 적용
- 성별 분리 모델 첫 시도 (M/F)
- LightGBM+Keyser, MLP 등 대안 모델 탐색

**05_Final_Model.ipynb** — 최종 모델
- 핵심 5변수: Exercise_Duration, BPM, Temp_diff, Age, Weight(lb)
- 성별 분리 (M/F 각각 학습)
- Poly(degree=2) + Ridge + 정수 반올림
- 성별별 degree/alpha 그리드 OOF 탐색
- **최종 OOF RMSE: 0.134 / LB: 0.12**
