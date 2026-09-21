# Mass–Spring–Damper Cube + Reinforcement Learning 수업 가이드

## 1. 수업 개요
**주제:** Mass–Spring–Damper로 3D cube를 만들고 PPO 강화학습으로 locomotion 학습  
**대상:** 딥러닝/강화학습 입문을 수강한 학부 3–4학년 또는 대학원 초급  
**권장 시간:** 3시간 수업 + 2시간 과제  
**도구:** Python, NumPy, PyTorch, Gymnasium, Matplotlib

### 학습 목표
수업이 끝나면 학생은 다음을 할 수 있다.
1. 질점-스프링-댐퍼 시스템의 운동 방정식을 코드로 구현한다.
2. 8개의 질점과 여러 스프링으로 cube를 구성한다.
3. 바닥 접촉, 중력, damping을 포함하는 간단한 물리 시뮬레이터를 만든다.
4. 물리 시뮬레이터를 강화학습 환경으로 정의한다.
5. observation / action / reward가 학습에 미치는 영향을 설명한다.
6. PPO actor-critic을 구현하고 학습 곡선을 해석한다.
7. reward shaping과 curriculum의 장단점을 실험한다.

---

## 2. 핵심 아이디어

### 2.1 질점 운동
질점 i의 위치와 속도는
- x_i ∈ R^3
- v_i ∈ R^3

뉴턴 법칙:
m_i a_i = ΣF_i

적분:
v_i(t+Δt) = v_i(t) + a_i Δt
x_i(t+Δt) = x_i(t) + v_i(t+Δt) Δt

수업에서는 이해를 위해 semi-implicit Euler를 사용한다.

### 2.2 Spring + Damper
두 질점 i, j를 연결하는 스프링의 길이:
L = ||x_j - x_i||

방향:
n = (x_j - x_i) / (L + ε)

Hooke force:
F_s = k (L - L0) n

상대속도의 스프링 방향 성분:
v_rel = (v_j - v_i) · n

Damper:
F_d = c v_rel n

두 질점에는 작용/반작용으로 반대 부호의 힘을 적용한다.

### 2.3 Cube
정육면체의 8개 vertex를 질점으로 사용한다.

단순히 12개의 edge spring만 쓰면 쉽게 찌그러질 수 있으므로 다음 연결을 사용한다.
- edge 12개
- face diagonal 12개
- body diagonal 4개

총 28개 spring.

### 2.4 Actuation
행동 a_s ∈ [-1, 1]이 선택된 actuator spring의 rest length를 바꾼다.

L0,s(t) = L0,s,base × (1 + A a_s)

A는 최대 수축/이완 비율이다. 예: A=0.20.

즉 agent는 직접 힘을 주는 것이 아니라 '근육처럼' 스프링의 자연 길이를 바꾼다.

---

## 3. 강화학습 문제 정의

### Observation
기본안:
- 모든 질점의 중심질량 기준 상대 위치: 8 × 3
- 모든 질점 속도: 8 × 3
- 중심질량 높이 1
- 중심질량 속도 3

총 52차원.

중심 위치의 절대 x,y를 제거하면 translation invariance를 얻을 수 있다.

### Action
선택된 actuator spring N_a개 각각에 대해
a ∈ [-1,1]^N_a

입문 수업에서는 12개 edge spring을 actuator로 두는 것을 권장한다.

### Reward
한 step의 전진량:
r_forward = x_COM(t+1) - x_COM(t)

에너지 비용:
r_energy = mean(a^2)

자세 안정성:
r_height = max(0, h_target - z_COM)

예:
r = 10 Δx_COM - 0.01 mean(a^2) - 0.5 max(0, 0.25-z_COM)

### Episode 종료
- 시간 제한 도달 → truncated
- cube가 지나치게 아래로 침투하거나 수치 폭발 → terminated

---

## 4. 수업 진행안 (180분)

| 시간 | 내용 | 학생 활동 |
|---|---|---|
| 0–15분 | 문제 소개 | 영상/그림으로 soft robot 개념 이해 |
| 15–40분 | mass–spring–damper 이론 | force 식 직접 유도 |
| 40–70분 | cube simulator 구현 | spring force와 integration 완성 |
| 70–85분 | 디버깅 | gravity/contact test |
| 85–95분 | 휴식 | |
| 95–120분 | Gymnasium 환경 설계 | obs/action/reward 토론 |
| 120–145분 | PPO actor-critic | clipped objective 설명 |
| 145–170분 | 학습 실습 | reward curve 확인 |
| 170–180분 | 실험 설계 | 과제 설명 |

---

## 5. 강의용 설명 포인트

### 왜 Euler가 아니라 semi-implicit Euler인가?
위치를 업데이트할 때 새 속도를 사용한다. 동일한 계산량으로 단순 explicit Euler보다 기계 시스템에서 대체로 안정적이다.

### 왜 damping이 필요한가?
스프링만 있으면 에너지가 계속 진동 형태로 남는다. damping은 에너지를 소산시켜 수치적으로도 안정적인 동작을 만든다.

### 왜 edge spring만으로 부족한가?
정육면체의 면이 shear 형태로 쉽게 변형된다. face/body diagonal이 구조적 강성을 추가한다.

### 왜 절대 위치보다 상대 위치를 관측하는가?
현재 x=0인지 x=10인지보다 cube의 '형상'과 속도가 행동 결정에 더 중요하다. 불필요한 상태 자유도를 제거하면 학습이 쉬워질 수 있다.

### 왜 reward shaping이 필요한가?
최종 위치만 주는 sparse reward보다 전진량을 매 step 제공하면 credit assignment가 쉬워진다. 다만 shaping이 지나치면 의도하지 않은 행동을 학습할 수 있다.

---

## 6. PPO 핵심 수식

정책 비율:
r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)

Clipped surrogate:
L_clip = E[min(r_t A_t, clip(r_t,1-ε,1+ε)A_t)]

전체 loss의 전형적 형태:
Loss = -L_clip + c_v L_value - c_e H(π)

- L_value: critic MSE
- H: entropy
- ε: clip range

GAE:
δ_t = r_t + γV(s_{t+1}) - V(s_t)

A_t = δ_t + γλ δ_{t+1} + ...

---

## 7. 실험 과제

### 필수 과제
1. edge spring만 사용한 cube와 diagonal spring까지 사용한 cube를 비교한다.
2. damping coefficient c = {0.2, 1.0, 3.0}에서 안정성을 비교한다.
3. reward에서 energy penalty를 제거했을 때 행동을 관찰한다.
4. 최소 3개 random seed의 학습 curve를 제시한다.
5. best policy rollout을 시각화하고 동작 원리를 설명한다.

### 선택 과제
- action을 spring 길이가 아니라 stiffness k로 정의
- 목표 방향을 observation에 추가해 2D navigation
- terrain height 변화
- domain randomization: mass, k, friction을 episode마다 변경
- curriculum: 낮은 중력 → 정상 중력
- recurrent policy로 변경

---

## 8. 토론 질문
1. observation에서 velocity를 제거하면 Markov property가 유지되는가?
2. spring stiffness가 매우 커지면 왜 작은 Δt가 필요한가?
3. reward에 COM forward displacement만 주면 어떤 reward hacking이 가능한가?
4. 구조 topology 자체를 학습 대상으로 만들 수 있는가?
5. simulator와 실제 soft robot 사이 sim-to-real gap의 원인은 무엇인가?

---

## 9. 평가 루브릭 (100점)
- 물리 시뮬레이터 정확성 25
- Gymnasium 환경 설계 15
- PPO 구현/학습 20
- 실험 설계 및 비교 20
- 그래프/시각화 10
- 해석 및 보고서 10

---

## 10. 설치
```bash
pip install numpy matplotlib torch gymnasium
```

Notebook: `mass_spring_cube_rl_tutorial.ipynb`