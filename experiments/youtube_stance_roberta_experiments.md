# YouTube 법·정책 댓글 Encoder 찬반 분류 실험 원장

> 완료된 실행의 `run_summary.json`을 확인한 뒤 성공한 실험만 수동으로 기록합니다.
> LLM silver 라벨 기준 지표는 사람 gold 라벨 기준 실제 성능으로 해석하지 않습니다.

## 전체 실험 요약

| 실험 | 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---:|---:|---:|---:|
| E01 | RoBERTa 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · sqrt_balanced 가중 손실 | 검증(dev) | 0.6901 | 0.6038 | 0.6061 | 0.6044 |
| E01 | RoBERTa 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · sqrt_balanced 가중 손실 | 테스트(test) | 0.7624 | 0.7096 | 0.7681 | 0.7260 |
| E02 | RoBERTa 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 5-fold 내부 검증 평균±표준편차 | 0.7251 ± 0.0386 | 0.6615 ± 0.0392 | 0.6576 ± 0.0626 | 0.6532 ± 0.0508 |
| E02 | RoBERTa 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 임계값 설정용 검증(dev) | 0.7217 | 0.6677 | 0.6095 | 0.6297 |
| E02 | RoBERTa 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 최종 테스트(test) | 0.7873 | 0.7374 | 0.7472 | 0.7410 |
| E03 | KcELECTRA 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · sqrt_balanced 가중 손실 | 검증(dev) | 0.7145 | 0.6454 | 0.6092 | 0.6240 |
| E03 | KcELECTRA 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · sqrt_balanced 가중 손실 | 테스트(test) | 0.7826 | 0.7230 | 0.7248 | 0.7228 |
| E04 | KcELECTRA 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 5-fold 내부 검증 평균±표준편차 | 0.7009 ± 0.0509 | 0.6364 ± 0.0747 | 0.6201 ± 0.0697 | 0.6151 ± 0.0614 |
| E04 | KcELECTRA 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 임계값 설정용 검증(dev) | 0.6643 | 0.5854 | 0.6054 | 0.5926 |
| E04 | KcELECTRA 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 최종 테스트(test) | 0.7609 | 0.7025 | 0.7325 | 0.7101 |
| E05 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 5-fold 내부 검증 평균±표준편차 | 0.8423 ± 0.0313 | 0.7445 ± 0.0935 | 0.6406 ± 0.0792 | 0.6611 ± 0.0827 |
| E05 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 임계값 설정용 검증(dev) | 0.8216 | 0.4108 | 0.5000 | 0.4510 |
| E05 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 최종 테스트(test) | 0.8148 | 0.6834 | 0.6600 | 0.6699 |
| E06 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 단일 학습 · dev 최고 epoch 선택 · sqrt_balanced 가중 손실 | 검증(dev) | 0.7552 | 0.6275 | 0.6644 | 0.6383 |
| E06 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 단일 학습 · dev 최고 epoch 선택 · sqrt_balanced 가중 손실 | 최종 테스트(test) | 0.7984 | 0.6905 | 0.7458 | 0.7083 |
| E07 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 3-fold CV · fold당 5 epoch · sqrt_balanced 가중 손실 | 3-fold 내부 검증 평균±표준편차 | 0.8363 ± 0.0365 | 0.6982 ± 0.1228 | 0.6347 ± 0.1049 | 0.6482 ± 0.1249 |
| E07 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 3-fold CV · fold당 5 epoch · sqrt_balanced 가중 손실 | 임계값 설정용 검증(dev) | 0.5539 | 0.4978 | 0.4964 | 0.4645 |
| E07 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 3-fold CV · fold당 5 epoch · sqrt_balanced 가중 손실 | 최종 테스트(test) | 0.6358 | 0.5994 | 0.6638 | 0.5756 |
| E08 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 단일 학습 · train 찬성:반대 2:1 오버샘플링 · 5 epoch 고정 · class weight 없음 | 검증(dev) | 0.8506 | 0.7463 | 0.7680 | 0.7562 |
| E08 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 단일 학습 · train 찬성:반대 2:1 오버샘플링 · 5 epoch 고정 · class weight 없음 | 최종 테스트(test) | 0.8498 | 0.7502 | 0.7773 | 0.7622 |
| E08-01 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 고정 split에서 5-seed 단일 학습 · train 찬성:반대 2:1 오버샘플링 · seed별 5 epoch 고정 · class weight 없음 | 검증(dev) 5-seed 평균±표준편차 | 0.8544 ± 0.0188 | 0.7504 ± 0.0347 | 0.7430 ± 0.0563 | 0.7451 ± 0.0457 |
| E08-01 | KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 고정 split에서 5-seed 단일 학습 · train 찬성:반대 2:1 오버샘플링 · seed별 5 epoch 고정 · class weight 없음 | 최종 테스트(test) 5-seed 평균±표준편차 | 0.8379 ± 0.0174 | 0.7311 ± 0.0274 | 0.7421 ± 0.0345 | 0.7353 ± 0.0297 |

## 기록 항목

- 실행 ID와 시작·종료 시각
- 입력 데이터 SHA-256과 train/dev/test 구성
- 모델·시드·하이퍼파라미터·GPU 환경
- Accuracy, Macro Precision/Recall/F1, 클래스별 지표
- dev/test 혼동행렬과 최적 체크포인트
- 모델·예측·분할표·JSON 산출물 경로

## 20260804_125646_KST_seed42 — 완료

### 실험 요약

| 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| RoBERTa 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · sqrt_balanced 가중 손실 | 검증(dev) | 0.6901 | 0.6038 | 0.6061 | 0.6044 |
| RoBERTa 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · sqrt_balanced 가중 손실 | 테스트(test) | 0.7624 | 0.7096 | 0.7681 | 0.7260 |

- 실험명: `youtube-stance-klue-roberta-base`
- 시작: 2026-08-04T12:56:49+09:00
- 종료: 2026-08-04T12:58:20+09:00
- 모델: `klue/roberta-base`
- 출력 폴더: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_base/20260804_125646_KST_seed42`
- 메모: E01

### 한줄 요약

dev Macro-F1 `0.6044`, test Macro-F1 `0.7260`; LLM silver 라벨에 대한 일치 성능입니다.

### 데이터

- 입력 SHA-256: `5625c4b049f3dc142081b8a329e1c5df43c4e6ddd751e947c7a754dd83d4c713`
- 원본/선택/제외 행: 7558 / 6951 / 607
- 선택 영상 수: 131
- `needs_review=true` 포함: False

| split | 행 | 영상 | 누수 방지 그룹 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|---:|---:|---:|
| train | 5610 | 108 | 94 | 739 | 3295 | 1576 |
| dev | 697 | 12 | 11 | 77 | 415 | 205 |
| test | 644 | 11 | 10 | 85 | 365 | 194 |

### 주요 설정

| 항목 | 값 |
|---|---|
| seed | 42 |
| text mode | comment_context |
| max length | 256 |
| epochs | 5.0 |
| train batch | 16 |
| eval batch | 32 |
| gradient accumulation | 1 |
| learning rate | 2e-05 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| warmup steps | 176 |
| class weighting | sqrt_balanced |

### 결과

| 평가 데이터 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| dev | 0.6901 | 0.6038 | 0.6061 | 0.6044 |
| test | 0.7624 | 0.7096 | 0.7681 | 0.7260 |

| 평가 데이터 | 클래스 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| dev | 찬성 | 0.3765 | 0.4156 | 0.3951 |
| dev | 반대 | 0.7613 | 0.7687 | 0.7650 |
| dev | 중립 | 0.6736 | 0.6341 | 0.6533 |
| test | 찬성 | 0.5037 | 0.8000 | 0.6182 |
| test | 반대 | 0.8917 | 0.7671 | 0.8247 |
| test | 중립 | 0.7333 | 0.7371 | 0.7352 |

- train loss: 0.4917
- train runtime(초): 76.28
- best checkpoint: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_base/20260804_125646_KST_seed42/checkpoints/checkpoint-702`
- best dev metric: 0.6044

#### Dev 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|
| 찬성 | 32 | 38 | 7 |
| 반대 | 40 | 319 | 56 |
| 중립 | 13 | 62 | 130 |

#### Test 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|
| 찬성 | 68 | 8 | 9 |
| 반대 | 42 | 280 | 43 |
| 중립 | 25 | 26 | 143 |

### 환경 및 산출물

- Python: `3.12.13`
- PyTorch: `2.12.1+cu130`
- Transformers: `5.12.1`
- CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5070`
- 모델: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_base/20260804_125646_KST_seed42/model`
- JSON 요약: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_base/20260804_125646_KST_seed42/run_summary.json`
- 분할표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_base/20260804_125646_KST_seed42/split_manifest.csv`
- dev 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_base/20260804_125646_KST_seed42/dev_predictions.csv`
- test 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_base/20260804_125646_KST_seed42/test_predictions.csv`

> 주의: dev/test도 LLM silver 라벨입니다. 실제 성능 주장은 별도의 사람 gold 평가셋으로 확인해야 합니다.

## 20260804_131503_KST_E02_seed42 — 완료

### 실험 요약

| 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| RoBERTa 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 5-fold 내부 검증 평균±표준편차 | 0.7251 ± 0.0386 | 0.6615 ± 0.0392 | 0.6576 ± 0.0626 | 0.6532 ± 0.0508 |
| RoBERTa 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 임계값 설정용 검증(dev) | 0.7217 | 0.6677 | 0.6095 | 0.6297 |
| RoBERTa 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 최종 테스트(test) | 0.7873 | 0.7374 | 0.7472 | 0.7410 |

- 실험명: `youtube-stance-klue-roberta-base-cv5-e3`
- 시작: 2026-08-04T13:15:07+09:00
- 종료: 2026-08-04T13:19:52+09:00
- 모델: `klue/roberta-base`
- 출력 폴더: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_cv5_e3/20260804_131503_KST_E02_seed42`
- 메모: E02

### 한줄 요약

5-fold 내부 검증 Macro-F1 `0.6532 ± 0.0508`, 임계값 설정용 dev Macro-F1 `0.6297`, 최종 test Macro-F1 `0.7410`; LLM silver 라벨에 대한 일치 성능입니다.

### 데이터

- 입력 SHA-256: `5625c4b049f3dc142081b8a329e1c5df43c4e6ddd751e947c7a754dd83d4c713`
- 원본/선택/제외 행: 7558 / 6951 / 607
- 선택 영상 수: 131
- `needs_review=true` 포함: False

| split | 행 | 영상 | 누수 방지 그룹 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|---:|---:|---:|
| train | 5610 | 108 | 94 | 739 | 3295 | 1576 |
| dev | 697 | 12 | 11 | 77 | 415 | 205 |
| test | 644 | 11 | 10 | 85 | 365 | 194 |

### 주요 설정

| 항목 | 값 |
|---|---|
| seed | 42 |
| folds | 5 |
| text mode | comment_context |
| max length | 256 |
| epochs | 3.0 |
| train batch | 16 |
| eval batch | 32 |
| gradient accumulation | 1 |
| learning rate | 2e-05 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| class weighting | sqrt_balanced |

### 임계값 설정

- 검증(dev) 10%에서만 Macro-F1 기준으로 설정
- 결정 규칙: `argmax(probability / class_threshold)`
- 찬성/반대/중립 임계값: `0.6000` / `0.5800` / `0.5000`
- 테스트 평가 횟수: 1

### 결과

| 평가 데이터 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 5-fold 내부 검증 평균±표준편차 | 0.7251 ± 0.0386 | 0.6615 ± 0.0392 | 0.6576 ± 0.0626 | 0.6532 ± 0.0508 |
| 임계값 설정용 dev | 0.7217 | 0.6677 | 0.6095 | 0.6297 |
| 최종 test | 0.7873 | 0.7374 | 0.7472 | 0.7410 |

| 평가 데이터 | 클래스 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| 5-fold 내부 검증 평균±표준편차 | 찬성 | 0.5096 ± 0.0507 | 0.5144 ± 0.2033 | 0.4967 ± 0.1155 |
| 5-fold 내부 검증 평균±표준편차 | 반대 | 0.7950 ± 0.0588 | 0.8016 ± 0.0352 | 0.7972 ± 0.0361 |
| 5-fold 내부 검증 평균±표준편차 | 중립 | 0.6800 ± 0.0555 | 0.6569 ± 0.0900 | 0.6656 ± 0.0625 |
| 임계값 설정용 dev | 찬성 | 0.5625 | 0.3506 | 0.4320 |
| 임계값 설정용 dev | 반대 | 0.7533 | 0.8241 | 0.7871 |
| 임계값 설정용 dev | 중립 | 0.6872 | 0.6537 | 0.6700 |
| 최종 test | 찬성 | 0.5876 | 0.6706 | 0.6264 |
| 최종 test | 반대 | 0.8424 | 0.8493 | 0.8458 |
| 최종 test | 중립 | 0.7821 | 0.7216 | 0.7507 |

- train loss: 0.5930
- train runtime(초): 239.56
- best checkpoint: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_cv5_e3/20260804_131503_KST_E02_seed42/fold_3/checkpoints/checkpoint-819`
- best dev metric: 0.7080

#### Dev 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|
| 찬성 | 27 | 42 | 8 |
| 반대 | 20 | 342 | 53 |
| 중립 | 1 | 70 | 134 |

#### Test 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|
| 찬성 | 57 | 20 | 8 |
| 반대 | 24 | 310 | 31 |
| 중립 | 16 | 38 | 140 |

### 환경 및 산출물

- Python: `3.12.13`
- PyTorch: `2.12.1+cu130`
- Transformers: `5.12.1`
- CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5070`
- 모델: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_cv5_e3/20260804_131503_KST_E02_seed42/fold_*/model`
- JSON 요약: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_cv5_e3/20260804_131503_KST_E02_seed42/run_summary.json`
- 분할표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_cv5_e3/20260804_131503_KST_E02_seed42/split_manifest.csv`
- dev 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_cv5_e3/20260804_131503_KST_E02_seed42/dev_threshold_predictions.csv`
- test 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_klue_roberta_cv5_e3/20260804_131503_KST_E02_seed42/test_ensemble_predictions.csv`

> 주의: dev/test도 LLM silver 라벨입니다. 실제 성능 주장은 별도의 사람 gold 평가셋으로 확인해야 합니다.
## 20260804_142836_KST_E03_seed42 — 완료

### 실험 요약

| 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| KcELECTRA 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · sqrt_balanced 가중 손실 | 검증(dev) | 0.7145 | 0.6454 | 0.6092 | 0.6240 |
| KcELECTRA 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · sqrt_balanced 가중 손실 | 테스트(test) | 0.7826 | 0.7230 | 0.7248 | 0.7228 |

- 실험명: `youtube-stance-kcelectra-base`
- 시작: 2026-08-04T14:28:40+09:00
- 종료: 2026-08-04T14:30:35+09:00
- 모델: `beomi/KcELECTRA-base`
- 출력 폴더: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_base/20260804_142836_KST_E03_seed42`
- 메모: E03

### 한줄 요약

dev Macro-F1 `0.6240`, test Macro-F1 `0.7228`; LLM silver 라벨에 대한 일치 성능입니다.

### 데이터

- 입력 SHA-256: `5625c4b049f3dc142081b8a329e1c5df43c4e6ddd751e947c7a754dd83d4c713`
- 원본/선택/제외 행: 7558 / 6951 / 607
- 선택 영상 수: 131
- `needs_review=true` 포함: False

| split | 행 | 영상 | 누수 방지 그룹 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|---:|---:|---:|
| train | 5610 | 108 | 94 | 739 | 3295 | 1576 |
| dev | 697 | 12 | 11 | 77 | 415 | 205 |
| test | 644 | 11 | 10 | 85 | 365 | 194 |

### 주요 설정

| 항목 | 값 |
|---|---|
| seed | 42 |
| folds | - |
| text mode | comment_context |
| token type IDs | True |
| max length | 256 |
| epochs | 5.0 |
| train batch | 16 |
| eval batch | 32 |
| gradient accumulation | 1 |
| learning rate | 2e-05 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| class weighting | sqrt_balanced |

### 결과

| 평가 데이터 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| dev | 0.7145 | 0.6454 | 0.6092 | 0.6240 |
| test | 0.7826 | 0.7230 | 0.7248 | 0.7228 |

| 평가 데이터 | 클래스 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| dev | 찬성 | 0.4915 | 0.3766 | 0.4265 |
| dev | 반대 | 0.7533 | 0.8169 | 0.7838 |
| dev | 중립 | 0.6915 | 0.6341 | 0.6616 |
| test | 찬성 | 0.5426 | 0.6000 | 0.5698 |
| test | 반대 | 0.8422 | 0.8630 | 0.8525 |
| test | 중립 | 0.7841 | 0.7113 | 0.7459 |

- train loss: 0.5012
- train runtime(초): 89.46
- best checkpoint: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_base/20260804_142836_KST_E03_seed42/checkpoints/checkpoint-1755`
- best dev metric: 0.6240

#### Dev 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|
| 찬성 | 29 | 39 | 9 |
| 반대 | 27 | 339 | 49 |
| 중립 | 3 | 72 | 130 |

#### Test 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|
| 찬성 | 51 | 18 | 16 |
| 반대 | 28 | 315 | 22 |
| 중립 | 15 | 41 | 138 |

### 환경 및 산출물

- Python: `3.12.13`
- PyTorch: `2.12.1+cu130`
- Transformers: `5.12.1`
- CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5070`
- 모델: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_base/20260804_142836_KST_E03_seed42/model`
- JSON 요약: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_base/20260804_142836_KST_E03_seed42/run_summary.json`
- 분할표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_base/20260804_142836_KST_E03_seed42/split_manifest.csv`
- dev 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_base/20260804_142836_KST_E03_seed42/dev_predictions.csv`
- test 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_base/20260804_142836_KST_E03_seed42/test_predictions.csv`

> 주의: dev/test도 LLM silver 라벨입니다. 실제 성능 주장은 별도의 사람 gold 평가셋으로 확인해야 합니다.
## 20260804_144213_KST_E04_seed42 — 완료

### 실험 요약

| 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| KcELECTRA 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 5-fold 내부 검증 평균±표준편차 | 0.7009 ± 0.0509 | 0.6364 ± 0.0747 | 0.6201 ± 0.0697 | 0.6151 ± 0.0614 |
| KcELECTRA 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 임계값 설정용 검증(dev) | 0.6643 | 0.5854 | 0.6054 | 0.5926 |
| KcELECTRA 전체 파인튜닝 · LLM silver 3분류 · 댓글+법률·영상 문맥 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 최종 테스트(test) | 0.7609 | 0.7025 | 0.7325 | 0.7101 |

- 실험명: `youtube-stance-kcelectra-base-cv5-e3`
- 시작: 2026-08-04T14:42:15+09:00
- 종료: 2026-08-04T14:46:44+09:00
- 모델: `beomi/KcELECTRA-base`
- 출력 폴더: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_cv5_e3/20260804_144213_KST_E04_seed42`
- 메모: E04

### 한줄 요약

5-fold 내부 검증 Macro-F1 `0.6151 ± 0.0614`, 임계값 설정용 dev Macro-F1 `0.5926`, 최종 test Macro-F1 `0.7101`; LLM silver 라벨에 대한 일치 성능입니다.

### 데이터

- 입력 SHA-256: `5625c4b049f3dc142081b8a329e1c5df43c4e6ddd751e947c7a754dd83d4c713`
- 원본/선택/제외 행: 7558 / 6951 / 607
- 선택 영상 수: 131
- `needs_review=true` 포함: False

| split | 행 | 영상 | 누수 방지 그룹 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|---:|---:|---:|
| train | 5610 | 108 | 94 | 739 | 3295 | 1576 |
| dev | 697 | 12 | 11 | 77 | 415 | 205 |
| test | 644 | 11 | 10 | 85 | 365 | 194 |

### 주요 설정

| 항목 | 값 |
|---|---|
| seed | 42 |
| folds | 5 |
| text mode | comment_context |
| token type IDs | True |
| max length | 256 |
| epochs | 3.0 |
| train batch | 16 |
| eval batch | 32 |
| gradient accumulation | 1 |
| learning rate | 2e-05 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| class weighting | sqrt_balanced |

### 임계값 설정

- 검증(dev) 10%에서만 Macro-F1 기준으로 설정
- 결정 규칙: `argmax(probability / class_threshold)`
- 찬성/반대/중립 임계값: `0.3400` / `0.8000` / `0.5000`
- 테스트 평가 횟수: 1

### 결과

| 평가 데이터 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 5-fold 내부 검증 평균±표준편차 | 0.7009 ± 0.0509 | 0.6364 ± 0.0747 | 0.6201 ± 0.0697 | 0.6151 ± 0.0614 |
| 임계값 설정용 dev | 0.6643 | 0.5854 | 0.6054 | 0.5926 |
| 최종 test | 0.7609 | 0.7025 | 0.7325 | 0.7101 |

| 평가 데이터 | 클래스 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| 5-fold 내부 검증 평균±표준편차 | 찬성 | 0.4455 ± 0.1521 | 0.4475 ± 0.2304 | 0.4187 ± 0.1509 |
| 5-fold 내부 검증 평균±표준편차 | 반대 | 0.7718 ± 0.0585 | 0.7986 ± 0.1033 | 0.7802 ± 0.0515 |
| 5-fold 내부 검증 평균±표준편차 | 중립 | 0.6920 ± 0.0662 | 0.6142 ± 0.0836 | 0.6464 ± 0.0492 |
| 임계값 설정용 dev | 찬성 | 0.3762 | 0.4935 | 0.4270 |
| 임계값 설정용 dev | 반대 | 0.7562 | 0.7325 | 0.7442 |
| 임계값 설정용 dev | 중립 | 0.6237 | 0.5902 | 0.6065 |
| 최종 test | 찬성 | 0.4597 | 0.6706 | 0.5455 |
| 최종 test | 반대 | 0.8513 | 0.8000 | 0.8249 |
| 최종 test | 중립 | 0.7966 | 0.7268 | 0.7601 |

- train loss: 0.7342
- train runtime(초): 223.91
- best checkpoint: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_cv5_e3/20260804_144213_KST_E04_seed42/fold_3/checkpoints/checkpoint-819`
- best dev metric: 0.7089

#### Dev 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|
| 찬성 | 38 | 30 | 9 |
| 반대 | 47 | 304 | 64 |
| 중립 | 16 | 68 | 121 |

#### Test 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 | 중립 |
|---|---:|---:|---:|
| 찬성 | 57 | 23 | 5 |
| 반대 | 42 | 292 | 31 |
| 중립 | 25 | 28 | 141 |

### 환경 및 산출물

- Python: `3.12.13`
- PyTorch: `2.12.1+cu130`
- Transformers: `5.12.1`
- CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5070`
- 모델: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_cv5_e3/20260804_144213_KST_E04_seed42/fold_*/model`
- JSON 요약: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_cv5_e3/20260804_144213_KST_E04_seed42/run_summary.json`
- 분할표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_cv5_e3/20260804_144213_KST_E04_seed42/split_manifest.csv`
- dev 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_cv5_e3/20260804_144213_KST_E04_seed42/dev_threshold_predictions.csv`
- test 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_cv5_e3/20260804_144213_KST_E04_seed42/test_ensemble_predictions.csv`

> 주의: dev/test도 LLM silver 라벨입니다. 실제 성능 주장은 별도의 사람 gold 평가셋으로 확인해야 합니다.

## 20260804_162221_KST_E05_seed42 — 완료

### 실험 요약

| 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 5-fold 내부 검증 평균±표준편차 | 0.8423 ± 0.0313 | 0.7445 ± 0.0935 | 0.6406 ± 0.0792 | 0.6611 ± 0.0827 |
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 임계값 설정용 검증(dev) | 0.8216 | 0.4108 | 0.5000 | 0.4510 |
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 5-fold CV · fold당 3 epoch · sqrt_balanced 가중 손실 | 최종 테스트(test) | 0.8148 | 0.6834 | 0.6600 | 0.6699 |

- 실험명: `youtube-stance-kcelectra-binary-law-original-cv5-e3`
- 시작: 2026-08-04T16:22:23+09:00
- 종료: 2026-08-04T16:31:22+09:00
- 모델: `beomi/KcELECTRA-base`
- 출력 폴더: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv5_e3/20260804_162221_KST_E05_seed42`
- 메모: E05

### 한줄 요약

5-fold 내부 검증 Macro-F1 `0.6611 ± 0.0827`, 임계값 설정용 dev Macro-F1 `0.4510`, 최종 test Macro-F1 `0.6699`; 중립을 제외한 LLM silver 찬성/반대 라벨에 대한 일치 성능입니다.

### 데이터 및 피처

- 타겟 변환: `긍정 → 찬성`, `부정 → 반대`; 중립 제외
- 입력 SHA-256: `5625c4b049f3dc142081b8a329e1c5df43c4e6ddd751e947c7a754dd83d4c713`
- 원본/선택/제외 행: 7558 / 4976 / 2582
- 선택 라벨: 찬성 901 / 반대 4075
- 선택 영상 수: 129
- `needs_review=true` 포함: False
- 입력 피처: 댓글, 법률명, 일반/반어, 댓글 관련 조문·개정 이유 원문
- 타겟 라벨 및 영상 제목의 피처 포함: False
- 법률 원문 파일 수: 10
- 법률 원문 결합 SHA-256: `8362597555dbfdb5dcd297047e94a81cbd1c4a0296fb0ecd9b48bd27abefe9ad`
- 댓글별 관련 원문: 최대 3구간, 600자; 요약·의역 없이 원문 사용

| split | 행 | 영상 | 누수 방지 그룹 | 찬성 | 반대 |
|---|---:|---:|---:|---:|---:|
| train | 4008 | 102 | 96 | 726 | 3282 |
| dev | 482 | 10 | 9 | 86 | 396 |
| test | 486 | 17 | 17 | 89 | 397 |

### 주요 설정

| 항목 | 값 |
|---|---|
| seed | 42 |
| folds | 5 |
| text mode | comment_context |
| context mode | law_original |
| token type IDs | True |
| max length | 512 |
| epochs | 3.0 |
| train batch | 4 |
| eval batch | 8 |
| gradient accumulation | 4 |
| effective train batch | 16 |
| learning rate | 2e-05 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| class weighting | sqrt_balanced |

### 임계값 설정

- 검증(dev) 10%에서만 Macro-F1 기준으로 설정
- 결정 규칙: `P(찬성) >= 0.5600`이면 찬성, 아니면 반대
- 찬성/반대 임계값: `0.5600` / `0.4400`
- 테스트 평가 횟수: 1

### 결과

| 평가 데이터 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 5-fold 내부 검증 평균±표준편차 | 0.8423 ± 0.0313 | 0.7445 ± 0.0935 | 0.6406 ± 0.0792 | 0.6611 ± 0.0827 |
| 임계값 설정용 dev | 0.8216 | 0.4108 | 0.5000 | 0.4510 |
| 최종 test | 0.8148 | 0.6834 | 0.6600 | 0.6699 |

| 평가 데이터 | 클래스 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| 5-fold 내부 검증 평균±표준편차 | 찬성 | 0.6233 ± 0.1698 | 0.3248 ± 0.1677 | 0.4137 ± 0.1530 |
| 5-fold 내부 검증 평균±표준편차 | 반대 | 0.8657 ± 0.0298 | 0.9565 ± 0.0262 | 0.9084 ± 0.0180 |
| 임계값 설정용 dev | 찬성 | 0.0000 | 0.0000 | 0.0000 |
| 임계값 설정용 dev | 반대 | 0.8216 | 1.0000 | 0.9021 |
| 최종 test | 찬성 | 0.4933 | 0.4157 | 0.4512 |
| 최종 test | 반대 | 0.8735 | 0.9043 | 0.8886 |

- train loss: 2.0710
- 5개 fold 합산 train runtime(초): 462.65
- 전체 wall runtime(초): 539.44
- best fold: 2
- best checkpoint: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv5_e3/20260804_162221_KST_E05_seed42/fold_2/checkpoints/checkpoint-400`
- best fold validation Macro-F1: 0.7623

#### Dev 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 |
|---|---:|---:|
| 찬성 | 0 | 86 |
| 반대 | 0 | 396 |

#### Test 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 |
|---|---:|---:|
| 찬성 | 37 | 52 |
| 반대 | 38 | 359 |

### 해석상 주의

- dev에서는 모든 샘플을 반대로 예측해 찬성 Precision/Recall/F1이 모두 0입니다. 따라서 dev Macro-F1이 `0.4510`에 그쳤고 임계값의 일반화가 불안정합니다.
- test에서는 찬성 F1이 `0.4512`, 반대 F1이 `0.8886`으로 클래스 간 성능 차이가 큽니다.
- E04는 3분류·6951건·256토큰·기존 문맥이고, E05는 2분류·4976건·512토큰·법률 원문 문맥이므로 두 실험의 Macro-F1을 모델 효과만으로 직접 비교할 수 없습니다.

### 환경 및 산출물

- Python: `3.12.13`
- PyTorch: `2.12.1+cu130`
- Transformers: `5.12.1`
- CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5070`
- 모델: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv5_e3/20260804_162221_KST_E05_seed42/fold_*/model`
- JSON 요약: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv5_e3/20260804_162221_KST_E05_seed42/run_summary.json`
- 임계값: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv5_e3/20260804_162221_KST_E05_seed42/selected_thresholds.json`
- fold 지표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv5_e3/20260804_162221_KST_E05_seed42/fold_metrics.csv`
- 분할표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv5_e3/20260804_162221_KST_E05_seed42/split_manifest.csv`
- 피처 원문표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv5_e3/20260804_162221_KST_E05_seed42/feature_context_manifest.csv`
- dev 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv5_e3/20260804_162221_KST_E05_seed42/dev_threshold_predictions.csv`
- test 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv5_e3/20260804_162221_KST_E05_seed42/test_ensemble_predictions.csv`

> 주의: dev/test도 LLM silver 라벨입니다. 실제 성능 주장은 별도의 사람 gold 평가셋으로 확인해야 합니다.

## 20260804_164039_KST_E06_seed42 — 완료

### 실험 요약

| 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 단일 학습 · dev 최고 epoch 선택 · sqrt_balanced 가중 손실 | 검증(dev) | 0.7552 | 0.6275 | 0.6644 | 0.6383 |
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 단일 학습 · dev 최고 epoch 선택 · sqrt_balanced 가중 손실 | 최종 테스트(test) | 0.7984 | 0.6905 | 0.7458 | 0.7083 |

- 실험명: `youtube-stance-kcelectra-binary-law-original-single`
- 시작: 2026-08-04T16:40:42+09:00
- 종료: 2026-08-04T16:44:22+09:00
- 모델: `beomi/KcELECTRA-base`
- 출력 폴더: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single/20260804_164039_KST_E06_seed42`
- 메모: E06

### 한줄 요약

dev Macro-F1 `0.6383`, 최종 test Macro-F1 `0.7083`; E01 방식의 단일 학습으로 중립을 제외한 LLM silver 찬성/반대 라벨을 분류한 결과입니다.

### 데이터 및 피처

- 타겟 변환: `긍정 → 찬성`, `부정 → 반대`; 중립 제외
- 입력 SHA-256: `5625c4b049f3dc142081b8a329e1c5df43c4e6ddd751e947c7a754dd83d4c713`
- 원본/선택/제외 행: 7558 / 4976 / 2582
- 선택 라벨: 찬성 901 / 반대 4075
- 선택 영상 수: 129
- `needs_review=true` 포함: False
- 입력 피처: 댓글, 법률명, 일반/반어, 댓글 관련 조문·개정 이유 원문
- 타겟 라벨 및 영상 제목의 피처 포함: False
- 법률 원문 파일 수: 10
- 법률 원문 결합 SHA-256: `8362597555dbfdb5dcd297047e94a81cbd1c4a0296fb0ecd9b48bd27abefe9ad`
- 댓글별 관련 원문: 최대 3구간, 600자; 요약·의역 없이 원문 사용
- E05와 공통 열 기준 split 및 원문 피처 내용 동일: True

| split | 행 | 영상 | 누수 방지 그룹 | 찬성 | 반대 |
|---|---:|---:|---:|---:|---:|
| train | 4008 | 102 | 96 | 726 | 3282 |
| dev | 482 | 10 | 9 | 86 | 396 |
| test | 486 | 17 | 17 | 89 | 397 |

### 주요 설정

| 항목 | 값 |
|---|---|
| seed | 42 |
| 학습 방식 | train 80% 단일 학습, dev 최고 epoch 선택 및 조기 종료 |
| text mode | comment_context |
| context mode | law_original |
| token type IDs | True |
| max length | 512 |
| 최대 epochs | 5.0 |
| 실제 종료 epoch | 5.0 |
| early stopping patience | 2 |
| train batch | 4 |
| eval batch | 8 |
| gradient accumulation | 4 |
| effective train batch | 16 |
| learning rate | 2e-05 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| warmup steps | 126 |
| class weighting | sqrt_balanced |
| 찬성/반대 class weight | 1.3602 / 0.6398 |

### 결과

| 평가 데이터 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| dev | 0.7552 | 0.6275 | 0.6644 | 0.6383 |
| 최종 test | 0.7984 | 0.6905 | 0.7458 | 0.7083 |

| 평가 데이터 | 클래스 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| dev | 찬성 | 0.3689 | 0.5233 | 0.4327 |
| dev | 반대 | 0.8861 | 0.8056 | 0.8439 |
| 최종 test | 찬성 | 0.4646 | 0.6629 | 0.5463 |
| 최종 test | 반대 | 0.9164 | 0.8287 | 0.8704 |

- train loss: 1.6197
- train runtime(초): 184.22
- best checkpoint: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single/20260804_164039_KST_E06_seed42/checkpoints/checkpoint-1004`
- best dev Macro-F1: 0.6383

#### Dev 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 |
|---|---:|---:|
| 찬성 | 45 | 41 |
| 반대 | 77 | 319 |

#### Test 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 |
|---|---:|---:|
| 찬성 | 59 | 30 |
| 반대 | 68 | 329 |

### E05와 비교

- E05와 E06은 데이터 split, 법률 원문 피처, 모델, seed, max length가 동일합니다.
- 최종 test Macro-F1: E05 `0.6699` → E06 `0.7083` (`+0.0384`).
- 최종 test 찬성 F1: E05 `0.4512` → E06 `0.5463` (`+0.0951`).
- E05는 5-fold ensemble과 dev 임계값을 사용했고, E06은 단일 모델의 기본 argmax를 사용했으므로 차이는 학습·추론 방식 전체의 차이입니다.

### 환경 및 산출물

- Python: `3.12.13`
- PyTorch: `2.12.1+cu130`
- Transformers: `5.12.1`
- CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5070`
- 모델: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single/20260804_164039_KST_E06_seed42/model`
- JSON 요약: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single/20260804_164039_KST_E06_seed42/run_summary.json`
- 분할표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single/20260804_164039_KST_E06_seed42/split_manifest.csv`
- 피처 원문표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single/20260804_164039_KST_E06_seed42/feature_context_manifest.csv`
- trainer 로그: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single/20260804_164039_KST_E06_seed42/trainer_log_history.json`
- dev 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single/20260804_164039_KST_E06_seed42/dev_predictions.csv`
- test 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single/20260804_164039_KST_E06_seed42/test_predictions.csv`

> 주의: dev/test도 LLM silver 라벨입니다. 실제 성능 주장은 별도의 사람 gold 평가셋으로 확인해야 합니다.

## 20260804_165342_KST_E07_seed42 — 완료

### 실험 요약

| 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 3-fold CV · fold당 5 epoch · sqrt_balanced 가중 손실 | 3-fold 내부 검증 평균±표준편차 | 0.8363 ± 0.0365 | 0.6982 ± 0.1228 | 0.6347 ± 0.1049 | 0.6482 ± 0.1249 |
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 3-fold CV · fold당 5 epoch · sqrt_balanced 가중 손실 | 임계값 설정용 검증(dev) | 0.5539 | 0.4978 | 0.4964 | 0.4645 |
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 3-fold CV · fold당 5 epoch · sqrt_balanced 가중 손실 | 최종 테스트(test) | 0.6358 | 0.5994 | 0.6638 | 0.5756 |

- 실험명: `youtube-stance-kcelectra-binary-law-original-cv3-e5`
- 시작: 2026-08-04T16:53:46+09:00
- 종료: 2026-08-04T17:01:36+09:00
- 모델: `beomi/KcELECTRA-base`
- 출력 폴더: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/20260804_165342_KST_E07_seed42`
- 메모: E07

### 한줄 요약

3-fold 내부 검증 Macro-F1 `0.6482 ± 0.1249`, 임계값 설정용 dev Macro-F1 `0.4645`, 최종 test Macro-F1 `0.5756`; fold별 편차와 dev 임계값의 불안정성이 큰 결과입니다.

### 데이터 및 피처

- 타겟 변환: `긍정 → 찬성`, `부정 → 반대`; 중립 제외
- 입력 SHA-256: `5625c4b049f3dc142081b8a329e1c5df43c4e6ddd751e947c7a754dd83d4c713`
- 원본/선택/제외 행: 7558 / 4976 / 2582
- 선택 라벨: 찬성 901 / 반대 4075
- 선택 영상 수: 129
- `needs_review=true` 포함: False
- 입력 피처: 댓글, 법률명, 일반/반어, 댓글 관련 조문·개정 이유 원문
- 타겟 라벨 및 영상 제목의 피처 포함: False
- 법률 원문 파일 수: 10
- 법률 원문 결합 SHA-256: `8362597555dbfdb5dcd297047e94a81cbd1c4a0296fb0ecd9b48bd27abefe9ad`
- 댓글별 관련 원문: 최대 3구간, 600자; 요약·의역 없이 원문 사용
- E05와 train/dev/test split 및 원문 피처 내용 동일: True
- E05와 CV fold 할당 동일: False (E05 5-fold, E07 3-fold)

| split | 행 | 영상 | 누수 방지 그룹 | 찬성 | 반대 |
|---|---:|---:|---:|---:|---:|
| train | 4008 | 102 | 96 | 726 | 3282 |
| dev | 482 | 10 | 9 | 86 | 396 |
| test | 486 | 17 | 17 | 89 | 397 |

### 주요 설정

| 항목 | 값 |
|---|---|
| seed | 42 |
| folds | 3 |
| fold당 epochs | 5.0 |
| text mode | comment_context |
| context mode | law_original |
| token type IDs | True |
| max length | 512 |
| train batch | 4 |
| eval batch | 8 |
| gradient accumulation | 4 |
| effective train batch | 16 |
| learning rate | 2e-05 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| class weighting | sqrt_balanced |

### 임계값 설정

- 검증(dev) 10%에서만 Macro-F1 기준으로 설정
- 결정 규칙: `P(찬성) >= 0.2200`이면 찬성, 아니면 반대
- 찬성/반대 임계값: `0.2200` / `0.7800`
- 임계값 적용 전 dev Macro-F1: `0.4510` (모두 반대로 예측)
- 임계값 적용 후 dev Macro-F1: `0.4645`
- 테스트 평가 횟수: 1

### 결과

| 평가 데이터 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 3-fold 내부 검증 평균±표준편차 | 0.8363 ± 0.0365 | 0.6982 ± 0.1228 | 0.6347 ± 0.1049 | 0.6482 ± 0.1249 |
| 임계값 설정용 dev | 0.5539 | 0.4978 | 0.4964 | 0.4645 |
| 최종 test | 0.6358 | 0.5994 | 0.6638 | 0.5756 |

| 평가 데이터 | 클래스 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| 3-fold 내부 검증 평균±표준편차 | 찬성 | 0.5323 ± 0.2110 | 0.3185 ± 0.2120 | 0.3913 ± 0.2312 |
| 3-fold 내부 검증 평균±표준편차 | 반대 | 0.8642 ± 0.0360 | 0.9509 ± 0.0030 | 0.9052 ± 0.0190 |
| 임계값 설정용 dev | 찬성 | 0.1759 | 0.4070 | 0.2456 |
| 임계값 설정용 dev | 반대 | 0.8198 | 0.5859 | 0.6834 |
| 최종 test | 찬성 | 0.2944 | 0.7079 | 0.4158 |
| 최종 test | 반대 | 0.9044 | 0.6196 | 0.7354 |

- train loss: 1.9307
- 3개 fold 합산 train runtime(초): 407.11
- 전체 wall runtime(초): 470.12
- best fold: 1
- best checkpoint: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/20260804_165342_KST_E07_seed42/fold_1/checkpoints/checkpoint-504`
- best fold validation Macro-F1: 0.7519

#### Dev 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 |
|---|---:|---:|
| 찬성 | 35 | 51 |
| 반대 | 164 | 232 |

#### Test 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 |
|---|---:|---:|
| 찬성 | 63 | 26 |
| 반대 | 151 | 246 |

### E05·E06과 비교

- 최종 test Macro-F1: E05 `0.6699`, E06 `0.7083`, E07 `0.5756`.
- E07은 E05 대비 test Macro-F1 `-0.0943`, 찬성 F1 `-0.0354`입니다.
- E07의 내부 검증 Macro-F1 표준편차는 `0.1249`로 E05의 `0.0827`보다 큽니다.
- E05 대비 fold 수와 epoch 수를 동시에 변경했으므로 E07의 차이를 epoch 증가 효과나 fold 감소 효과 하나로 분리해 해석할 수 없습니다.

### 해석상 주의

- dev에서 선택한 낮은 찬성 임계값 `0.22`가 test에서 찬성 Recall을 `0.7079`로 높였지만, 반대 397건 중 151건을 찬성으로 오분류했습니다.
- 클래스 불균형과 fold별 찬성 성능 변동이 커 단일 seed 결과만으로 일반화 성능을 결론내리기 어렵습니다.

### 환경 및 산출물

- Python: `3.12.13`
- PyTorch: `2.12.1+cu130`
- Transformers: `5.12.1`
- CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5070`
- 모델: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/20260804_165342_KST_E07_seed42/fold_*/model`
- JSON 요약: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/20260804_165342_KST_E07_seed42/run_summary.json`
- 임계값: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/20260804_165342_KST_E07_seed42/selected_thresholds.json`
- fold 지표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/20260804_165342_KST_E07_seed42/fold_metrics.csv`
- 분할표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/20260804_165342_KST_E07_seed42/split_manifest.csv`
- 피처 원문표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/20260804_165342_KST_E07_seed42/feature_context_manifest.csv`
- dev 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/20260804_165342_KST_E07_seed42/dev_threshold_predictions.csv`
- test 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/20260804_165342_KST_E07_seed42/test_ensemble_predictions.csv`

> 주의: dev/test도 LLM silver 라벨입니다. 실제 성능 주장은 별도의 사람 gold 평가셋으로 확인해야 합니다.

## 20260804_170954_KST_E08_seed42 — 완료

### 실험 요약

| 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 단일 학습 · train 찬성:반대 2:1 오버샘플링 · 5 epoch 고정 · class weight 없음 | 검증(dev) | 0.8506 | 0.7463 | 0.7680 | 0.7562 |
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 단일 학습 · train 찬성:반대 2:1 오버샘플링 · 5 epoch 고정 · class weight 없음 | 최종 테스트(test) | 0.8498 | 0.7502 | 0.7773 | 0.7622 |

- 실험명: `youtube-stance-kcelectra-binary-law-original-single-oversample-s2-o1`
- 시작: 2026-08-04T17:09:57+09:00
- 종료: 2026-08-04T17:18:01+09:00
- 모델: `beomi/KcELECTRA-base`
- 출력 폴더: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1/20260804_170954_KST_E08_seed42`
- 메모: E08

### 한줄 요약

dev Macro-F1 `0.7562`, 최종 test Macro-F1 `0.7622`; train 찬성을 복원추출로 늘려 찬성:반대를 2:1로 맞춘 단일 5 epoch 실험이며 E05~E08 이진 실험 중 test Macro-F1이 가장 높습니다.

### 데이터 및 피처

- 타겟 변환: `긍정 → 찬성`, `부정 → 반대`; 중립 제외
- 입력 SHA-256: `5625c4b049f3dc142081b8a329e1c5df43c4e6ddd751e947c7a754dd83d4c713`
- 원본/선택/제외 행: 7558 / 4976 / 2582
- 선택 라벨: 찬성 901 / 반대 4075
- 선택 영상 수: 129
- `needs_review=true` 포함: False
- 입력 피처: 댓글, 법률명, 일반/반어, 댓글 관련 조문·개정 이유 원문
- 타겟 라벨 및 영상 제목의 피처 포함: False
- 법률 원문 파일 수: 10
- 법률 원문 결합 SHA-256: `8362597555dbfdb5dcd297047e94a81cbd1c4a0296fb0ecd9b48bd27abefe9ad`
- 댓글별 관련 원문: 최대 3구간, 600자; 요약·의역 없이 원문 사용
- E06과 원본 split 및 원문 피처 내용 동일: True

| split | 원본 행 | 영상 | 누수 방지 그룹 | 찬성 | 반대 | 오버샘플링 |
|---|---:|---:|---:|---:|---:|---|
| train | 4008 | 102 | 96 | 726 | 3282 | 적용 |
| dev | 482 | 10 | 9 | 86 | 396 | 미적용 |
| test | 486 | 17 | 17 | 89 | 397 | 미적용 |

### Train 오버샘플링

| 항목 | 값 |
|---|---:|
| 적용 전 train 행 | 4008 |
| 적용 전 찬성 / 반대 | 726 / 3282 |
| 추가 찬성 복제본 | 5838 |
| 적용 후 train 행 | 9846 |
| 적용 후 찬성 / 반대 | 6564 / 3282 |
| 실제 찬성:반대 비율 | 2.0000:1 |

- 방식: train 내부 찬성 샘플만 seed 42로 복원 무작위 추출
- dev/test 오버샘플링: False
- 샘플링 manifest 재계산 검증: 9846행, 찬성 6564, 반대 3282, 복제본 5838

### 주요 설정

| 항목 | 값 |
|---|---|
| seed | 42 |
| 학습 방식 | train 80% 단일 학습, 정확히 5 epoch 실행 |
| text mode | comment_context |
| context mode | law_original |
| token type IDs | True |
| max length | 512 |
| epochs | 5.0 |
| 실제 종료 epoch | 5.0 |
| early stopping | 비활성화 |
| train batch | 4 |
| eval batch | 8 |
| gradient accumulation | 4 |
| effective train batch | 16 |
| learning rate | 2e-05 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| estimated total steps | 3080 |
| warmup steps | 308 |
| class weighting | none |
| 찬성/반대 class weight | 1.0 / 1.0 |

### 결과

| 평가 데이터 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| dev | 0.8506 | 0.7463 | 0.7680 | 0.7562 |
| 최종 test | 0.8498 | 0.7502 | 0.7773 | 0.7622 |

| 평가 데이터 | 클래스 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| dev | 찬성 | 0.5729 | 0.6395 | 0.6044 |
| dev | 반대 | 0.9197 | 0.8965 | 0.9079 |
| 최종 test | 찬성 | 0.5784 | 0.6629 | 0.6178 |
| 최종 test | 반대 | 0.9219 | 0.8917 | 0.9065 |

- train loss: 0.6403
- train runtime(초): 447.10
- best checkpoint: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1/20260804_170954_KST_E08_seed42/checkpoints/checkpoint-3080`
- best dev Macro-F1: 0.7562

#### Dev 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 |
|---|---:|---:|
| 찬성 | 55 | 31 |
| 반대 | 41 | 355 |

#### Test 혼동행렬

| 실제 \ 예측 | 찬성 | 반대 |
|---|---:|---:|
| 찬성 | 59 | 30 |
| 반대 | 43 | 354 |

### E06과 비교

- E06과 E08은 원본 train/dev/test split, 법률 원문 피처, 모델, seed, max length가 동일합니다.
- 최종 test Macro-F1: E06 `0.7083` → E08 `0.7622` (`+0.0538`).
- 최종 test 찬성 F1: E06 `0.5463` → E08 `0.6178` (`+0.0715`).
- 최종 test 반대 F1: E06 `0.8704` → E08 `0.9065` (`+0.0362`).
- E08은 오버샘플링과 함께 class weighting을 `sqrt_balanced`에서 `none`으로 바꿨으므로 성능 차이를 오버샘플링 단독 효과로 해석할 수 없습니다.

### 해석상 주의

- 동일 찬성 댓글을 복제한 방식이므로 새로운 찬성 표현이 추가된 것은 아닙니다.
- E08의 dev/test도 LLM silver 라벨이며, 사람 gold 평가 전에는 실제 찬반 분류 성능으로 단정할 수 없습니다.

### 환경 및 산출물

- Python: `3.12.13`
- PyTorch: `2.12.1+cu130`
- Transformers: `5.12.1`
- CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5070`
- 모델: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1/20260804_170954_KST_E08_seed42/model`
- JSON 요약: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1/20260804_170954_KST_E08_seed42/run_summary.json`
- 분할표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1/20260804_170954_KST_E08_seed42/split_manifest.csv`
- 피처 원문표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1/20260804_170954_KST_E08_seed42/feature_context_manifest.csv`
- 오버샘플링표: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1/20260804_170954_KST_E08_seed42/training_sampling_manifest.csv`
- trainer 로그: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1/20260804_170954_KST_E08_seed42/trainer_log_history.json`
- dev 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1/20260804_170954_KST_E08_seed42/dev_predictions.csv`
- test 예측: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1/20260804_170954_KST_E08_seed42/test_predictions.csv`

> 주의: dev/test도 LLM silver 라벨입니다. 실제 성능 주장은 별도의 사람 gold 평가셋으로 확인해야 합니다.

## E08-01 — 5-seed 완료

### 실험 요약

| 학습 방식 | 평가셋 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 고정 split에서 5-seed 단일 학습 · train 찬성:반대 2:1 오버샘플링 · seed별 5 epoch 고정 · class weight 없음 | 검증(dev) 5-seed 평균±표준편차 | 0.8544 ± 0.0188 | 0.7504 ± 0.0347 | 0.7430 ± 0.0563 | 0.7451 ± 0.0457 |
| KcELECTRA 전체 파인튜닝 · LLM silver 2분류 · 댓글+법률명·일반/반어·관련 법률 원문 · 고정 split에서 5-seed 단일 학습 · train 찬성:반대 2:1 오버샘플링 · seed별 5 epoch 고정 · class weight 없음 | 최종 테스트(test) 5-seed 평균±표준편차 | 0.8379 ± 0.0174 | 0.7311 ± 0.0274 | 0.7421 ± 0.0345 | 0.7353 ± 0.0297 |

- 실험명: `youtube-stance-kcelectra-binary-law-e08-01-five-seeds`
- 실행 기간: 2026-08-04T17:45:07+09:00 ~ 2026-08-04T18:25:46+09:00
- 모델: `beomi/KcELECTRA-base`
- 학습 seed: `7`, `19`, `42`, `73`, `123`
- 고정 split seed: `42`
- 출력 루트: `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_e08_01_five_seeds`
- 메모: E08-01

### 한줄 요약

동일 split에서 5개 학습 seed의 최종 test Macro-F1은 `0.7353 ± 0.0297`이며, 범위는 `0.6884~0.7617`입니다. 반대 F1은 비교적 안정적이지만 찬성 F1은 `0.4852~0.6102`로 seed 영향을 더 크게 받았습니다.

### 완료 및 무결성 수동 검증

- `run_summary.json` 완료 상태: 5 / 5
- dev/test 예측 CSV 독립 재계산과 저장 지표 일치: 5 / 5
- 5개 run의 split manifest 동일: True (`934029e76e328b80e8650600c8a4b6d9813d07575c310618752035e65932cca7`)
- 5개 run의 원문 피처 manifest 동일: True (`a4f4aa75ec8bee83ba6d6f81e11de8feb049bf431ed4953581a1e8dd63b8832e`)
- `comment_id`, `video_id`, `comment_hash`, `leakage_group`의 split 간 중복: 0
- NFKC·대소문자·공백 정규화 댓글의 split 간 중복: 0
- seed별 오버샘플링 manifest: 원본 train 4008행이 정확히 1회씩 존재하고 찬성 복제본 5838행만 추가됨
- 오버샘플링 데이터의 dev/test 댓글 유입: 0

### 데이터 및 피처

- 타겟 변환: `긍정 → 찬성`, `부정 → 반대`; 중립 제외
- 입력 SHA-256: `5625c4b049f3dc142081b8a329e1c5df43c4e6ddd751e947c7a754dd83d4c713`
- 원본/선택/제외 행: 7558 / 4976 / 2582
- 선택 라벨: 찬성 901 / 반대 4075
- 선택 영상 수: 129
- `needs_review=true` 포함: False
- 입력 피처: 댓글, 법률명, 일반/반어, 댓글 관련 조문·개정 이유 원문
- 타겟 라벨 및 영상 제목의 피처 포함: False
- 법률 원문 파일 수: 10
- 법률 원문 결합 SHA-256: `8362597555dbfdb5dcd297047e94a81cbd1c4a0296fb0ecd9b48bd27abefe9ad`
- 댓글별 관련 원문: 최대 3구간, 600자; 요약·의역 없이 원문 사용

| split | 원본 행 | 영상 | 누수 방지 그룹 | 찬성 | 반대 | 오버샘플링 |
|---|---:|---:|---:|---:|---:|---|
| train | 4008 | 102 | 96 | 726 | 3282 | seed별 train에만 적용 |
| dev | 482 | 10 | 9 | 86 | 396 | 미적용 |
| test | 486 | 17 | 17 | 89 | 397 | 미적용 |

### Train 오버샘플링

| 항목 | seed별 값 |
|---|---:|
| 적용 전 train 행 | 4008 |
| 적용 전 찬성 / 반대 | 726 / 3282 |
| 추가 찬성 복제본 | 5838 |
| 적용 후 train 행 | 9846 |
| 적용 후 찬성 / 반대 | 6564 / 3282 |
| 실제 찬성:반대 비율 | 2.0000:1 |

- 방식: 각 학습 seed로 train 내부 찬성 샘플만 복원 무작위 추출
- dev/test 오버샘플링: False
- 5개 seed 모두 샘플링 manifest 수동 검증 통과

### 주요 설정

| 항목 | 값 |
|---|---|
| training seeds | 7, 19, 42, 73, 123 |
| split seed | 42 |
| 학습 방식 | 동일한 train/dev/test에서 seed별 단일 학습 |
| text mode | comment_context |
| context mode | law_original |
| token type IDs | True |
| max length | 512 |
| epochs | seed별 5.0 |
| 실제 종료 epoch | 전 seed 5.0 |
| early stopping | 비활성화 |
| train batch | 4 |
| eval batch | 8 |
| gradient accumulation | 4 |
| effective train batch | 16 |
| learning rate | 2e-05 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| estimated total steps | seed별 3080 |
| warmup steps | seed별 308 |
| class weighting | none |
| 찬성/반대 class weight | 1.0 / 1.0 |

### Seed별 결과

| seed | 평가 데이터 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---:|---|---:|---:|---:|---:|
| 7 | dev | 0.8299 | 0.7043 | 0.6689 | 0.6831 |
| 7 | test | 0.8210 | 0.6971 | 0.6812 | 0.6884 |
| 19 | dev | 0.8651 | 0.7691 | 0.7996 | 0.7826 |
| 19 | test | 0.8189 | 0.7093 | 0.7497 | 0.7250 |
| 42 | dev | 0.8610 | 0.7626 | 0.7743 | 0.7682 |
| 42 | test | 0.8580 | 0.7628 | 0.7605 | 0.7617 |
| 73 | dev | 0.8402 | 0.7258 | 0.6980 | 0.7100 |
| 73 | test | 0.8498 | 0.7494 | 0.7642 | 0.7563 |
| 123 | dev | 0.8755 | 0.7902 | 0.7740 | 0.7817 |
| 123 | test | 0.8416 | 0.7368 | 0.7548 | 0.7451 |
| 평균±표준편차 | dev | 0.8544 ± 0.0188 | 0.7504 ± 0.0347 | 0.7430 ± 0.0563 | 0.7451 ± 0.0457 |
| 평균±표준편차 | test | 0.8379 ± 0.0174 | 0.7311 ± 0.0274 | 0.7421 ± 0.0345 | 0.7353 ± 0.0297 |

### 클래스별 5-seed 평균

| 평가 데이터 | 클래스 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| dev | 찬성 | 0.5928 ± 0.0504 | 0.5698 ± 0.1171 | 0.5784 ± 0.0816 |
| dev | 반대 | 0.9080 ± 0.0223 | 0.9162 ± 0.0115 | 0.9119 ± 0.0103 |
| test | 찬성 | 0.5550 ± 0.0465 | 0.5910 ± 0.0739 | 0.5707 ± 0.0510 |
| test | 반대 | 0.9072 ± 0.0143 | 0.8932 ± 0.0208 | 0.8999 ± 0.0112 |

### Seed별 혼동행렬

실제 행·예측 열 기준이며 `찬성→찬성`, `찬성→반대`, `반대→찬성`, `반대→반대` 순서입니다.

| seed | 평가 데이터 | 찬성→찬성 | 찬성→반대 | 반대→찬성 | 반대→반대 |
|---:|---|---:|---:|---:|---:|
| 7 | dev | 36 | 50 | 32 | 364 |
| 7 | test | 41 | 48 | 39 | 358 |
| 19 | dev | 60 | 26 | 39 | 357 |
| 19 | test | 57 | 32 | 56 | 341 |
| 42 | dev | 55 | 31 | 36 | 360 |
| 42 | test | 54 | 35 | 34 | 363 |
| 73 | dev | 41 | 45 | 32 | 364 |
| 73 | test | 56 | 33 | 40 | 357 |
| 123 | dev | 53 | 33 | 27 | 369 |
| 123 | test | 55 | 34 | 43 | 354 |

### 학습 로그 및 최적 체크포인트

| seed | train loss | runtime(초) | best dev Macro-F1 | best checkpoint step |
|---:|---:|---:|---:|---:|
| 7 | 0.7632 | 442.94 | 0.6831 | 1848 |
| 19 | 0.7345 | 447.52 | 0.7826 | 1232 |
| 42 | 0.5932 | 443.88 | 0.7682 | 2464 |
| 73 | 0.7299 | 446.06 | 0.7100 | 1232 |
| 123 | 0.7531 | 446.18 | 0.7817 | 2464 |

### 안정성 해석

- test Macro-F1 표본 표준편차는 `0.0297`, 범위는 `0.0733`으로 사전 점검 기준인 표준편차 `≤0.05`, 범위 `≤0.10`을 충족합니다.
- 다만 test 찬성 F1 범위가 `0.4852~0.6102`이므로 소수 클래스 성능은 seed에 민감합니다.
- test Macro-F1 최고 seed는 42의 `0.7617`이지만, test 결과를 보고 seed를 선택하면 안 됩니다. dev Macro-F1 최고 seed는 19의 `0.7826`이고 해당 seed의 test Macro-F1은 `0.7250`입니다.
- 기존 E08 단일 seed test Macro-F1 `0.7622` 대비 E08-01 5-seed 평균은 `0.7353`으로 `-0.0269`입니다. 단일 seed 최고값보다 평균과 분산을 함께 보는 편이 타당합니다.
- 동일 찬성 댓글을 복제한 방식이므로 새로운 찬성 표현이 추가된 것은 아닙니다.

### 환경 및 산출물

- Python: `3.12.13`
- PyTorch: `2.12.1+cu130`
- Transformers: `5.12.1`
- CUDA: `13.0`
- GPU: `NVIDIA GeForce RTX 5070`

| seed | 실행 ID | 출력 폴더 |
|---:|---|---|
| 7 | `20260804_174503_KST_E08-01_seed7` | `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_e08_01_five_seeds/20260804_174503_KST_E08-01_seed7` |
| 19 | `20260804_175308_KST_E08-01_seed19` | `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_e08_01_five_seeds/20260804_175308_KST_E08-01_seed19` |
| 42 | `20260804_180119_KST_E08-01_seed42` | `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_e08_01_five_seeds/20260804_180119_KST_E08-01_seed42` |
| 73 | `20260804_180927_KST_E08-01_seed73` | `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_e08_01_five_seeds/20260804_180927_KST_E08-01_seed73` |
| 123 | `20260804_181737_KST_E08-01_seed123` | `/workspace/samuel/policy_risk_persona_simulator/outputs/youtube_stance_kcelectra_binary_law_e08_01_five_seeds/20260804_181737_KST_E08-01_seed123` |

각 출력 폴더에는 모델, `run_summary.json`, `split_manifest.csv`, `feature_context_manifest.csv`, `training_sampling_manifest.csv`, `trainer_log_history.json`, `dev_predictions.csv`, `test_predictions.csv`가 있습니다.

> 주의: dev/test도 LLM silver 라벨입니다. 실제 성능 주장은 별도의 사람 gold 평가셋으로 확인해야 합니다.
