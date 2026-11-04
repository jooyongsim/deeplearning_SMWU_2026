# DL Assignment 01 보고서

- 이름 / 학번:
- 실행 환경 (OS, Python, CPU): `runs/m0/check_env.json` 참고
- 발표 영상 (YouTube): <링크>   ·   발표 자료: `slides.pptx`

## 1. 시연 데이터 (M1, M2)

- 에피소드 수, 성공 수, 총 스텝 / 조종한 스텝 (`runs/m2/summary.json`)
- `runs/m2/demos.png` 그림과 설명: 어떤 전략으로 밀었는지
- 시연의 일관성과 품질에 대한 관찰 (idle 비율, 키보드 이산 행동, 실패 시연)

## 2. 모방학습 설계 (M3)

- 입력 특징 (`make_features`): 무엇을 썼고 왜 썼는지
- 관측 / 행동 horizon, 정규화, 신경망 구조, 손실 함수, 하이퍼파라미터
- train / val 분할: 왜 에피소드 단위로 나누는지

## 3. 학습 결과 (M3)

- 손실 곡선 (`runs/m3/train_log.csv` → 그림), 과적합 여부

## 4. 폐루프 평가 (M4)

| agent | success (50) | ever | median pos / rot / shape |
|---|---|---|---|
| zero | | | |
| bc | | | |

- 결과 영상 목록 (`runs/m4/bc/videos/`): 파일 이름, 시드, 성공 여부, 한 줄 설명
- 성공 사례와 실패 사례 각 1개 이상: 영상의 장면(캡처 + 시각)을 근거로 무엇이 잘못되었는지
- 학습 손실과 폐루프 성공률의 관계 (compounding error)

## 5. Ablation

- 바꾼 것 1개 이상 (시연 수, 특징, `--only-success`, horizon 등)과 결과 표

## 6. 결론과 개선 방향
