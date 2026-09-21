# 과제: Learning to Move a Mass–Spring Cube

## 목표
직접 구현한 mass–spring–damper cube를 PPO로 학습시켜 +x 방향으로 이동시킨다.

## 제출물
1. 실행 가능한 notebook 또는 Python 코드
2. 학습 curve
3. best policy rollout figure 또는 animation
4. 2–4쪽 실험 보고서

## 필수 실험
- baseline 설정
- damping 3개 비교
- energy penalty on/off 비교
- seed 3개 이상 비교

## 보고서 질문
1. 어떤 hyperparameter가 물리 안정성에 가장 큰 영향을 주었는가?
2. 어떤 reward 구성에서 가장 자연스러운 gait가 나타났는가?
3. 학습된 policy가 실제로 주기적인 gait를 만들었는가?
4. 학습이 실패한 run에서 critic loss, entropy, return은 어떤 패턴을 보였는가?
5. 구조적 inductive bias를 더 넣는다면 무엇을 추가하겠는가?

## 채점
- 구현 40
- 실험 30
- 해석 20
- 재현성 10