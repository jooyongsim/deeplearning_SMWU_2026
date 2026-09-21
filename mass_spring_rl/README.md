# Mass-Spring-Damper Soft Cube + Reinforcement Learning

2026 딥러닝 수업용 튜토리얼 모음입니다. 8개의 질점과 spring-damper 구조로 soft cube를 만들고, 접촉 모델과 PPO 강화학습을 이용해 locomotion 및 goal-pose 제어를 학습합니다.

## 학습 흐름

### 1. Basic non-Warp tutorial
`basic_nonwarp/mass_spring_cube_rl_tutorial.ipynb`

- NumPy 기반 mass-spring-damper physics
- 8 particles, 28 spring-damper links
  - 12 edges
  - 12 face diagonals
  - 4 body diagonals
- 12 edge springs를 actuator로 사용
- Gymnasium environment
- PyTorch PPO
- 기본 goal: +x 방향 locomotion

함께 제공되는 자료:
- `basic_nonwarp/instructor_guide.md`
- `basic_nonwarp/assignment.md`
- `basic_nonwarp/requirements.txt`

### 2. NVIDIA Warp / GPU tutorial
`warp_colab/` 폴더의 순서대로 진행합니다.

1. `01_Warp_basics.ipynb`
   - Warp kernel/JIT
   - `wp.vec3`
   - semi-implicit Euler
   - Warp ↔ PyTorch zero-copy

2. `02_mass_spring_cube.ipynb`
   - Warp 기반 single cube
   - spring/damper
   - gravity + ground contact
   - actuator rest-length control

3. `03_vectorized_2048_cubes.ipynb`
   - 2048 environments 병렬 시뮬레이션
   - GPU vectorization
   - throughput benchmark

4. `04_PPO_training.ipynb`
   - vectorized PPO
   - GAE
   - clipped PPO objective
   - Warp physics + PyTorch policy

Colab에서는 GPU runtime을 권장합니다.

### 3. Goal position + orientation control
`goal_pose/` 폴더에는 동일한 task의 non-Warp / Warp 구현이 있습니다.

- `01_goal_pose_nonwarp.ipynb`
- `02_goal_pose_warp.ipynb`

기본 목표:
- target COM position: `(0.55, 0.25, SIDE/2)`
- target orientation: `+60 deg yaw`

Soft body는 rigid-body pose가 유일하지 않으므로, orientation은 reference cube와 현재 vertex 사이의 **Kabsch best-fit rotation**으로 정의합니다.

Reward는 position error와 orientation error의 감소량, actuation energy, deformation penalty를 함께 사용합니다.

### 4. Table contact model
초기 예제의 penalty contact에서 확장하여 goal-pose 버전에서는 다음을 사용합니다.

- table plane: `z = 0`
- restitution
- Coulomb-like tangential friction
- simple time-of-impact / projection

이는 `real2sim-eval`의 velocity/impulse 기반 collision 구조를 수업용으로 단순화한 것입니다.

### 5. Comparison with real2sim-eval
`goal_pose/COMPARISON_real2sim.md`

비교 대상:
https://github.com/kywind/real2sim-eval

핵심 차이:

| Item | This tutorial | real2sim-eval / PhysTwin |
|---|---|---|
| particles | 8 cube vertices | dense point cloud |
| topology | fixed 28 links | radius/KD-tree neighbor graph |
| spring law | Hooke 또는 strain | strain-like `(L/L0 - 1)` |
| damping | axial dashpot | axial dashpot + global drag |
| contact | table restitution/friction | ground + mesh + optional self collision |
| actuation | rest-length control | robot/object interaction 중심 |

`real2sim-eval`의 기본 수치 파라미터는 particle spacing, graph density, mass, timestep이 다르므로 이 8-node cube에 그대로 복사하지 않고 식의 구조만 참고해 재튜닝합니다.

## Suggested class sequence

### 1회차
- mass-spring-damper mechanics
- basic NumPy simulator
- cube topology
- contact

### 2회차
- Gymnasium formulation
- PPO / Actor-Critic / GAE
- forward locomotion

### 3회차
- Warp basics
- GPU vectorization
- 2048 parallel environments

### 4회차 / project
- goal-conditioned RL
- position + orientation control
- contact model ablation
- comparison with Real2Sim / PhysTwin

## Suggested experiments

- edge-only vs. 28-link structure
- Hooke extension vs. strain spring
- damping coefficient sweep
- dashpot-only vs. dashpot + global drag
- penalty contact vs. restitution/friction contact
- position-only reward vs. position+orientation reward
- randomized goal pose
- 64 / 256 / 1024 / 2048 / 4096 vectorized environments

## Dependencies

Basic version:
```bash
pip install numpy matplotlib torch gymnasium
```

Warp version:
```bash
pip install warp-lang torch numpy matplotlib
```
