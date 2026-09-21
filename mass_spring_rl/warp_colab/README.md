# Warp Mass–Spring Cube RL — Colab 4-Notebook Course

권장 실행 순서:

1. `01_Warp_basics.ipynb`
   - Warp kernel/JIT
   - vec3
   - semi-implicit Euler
   - PyTorch zero-copy interoperability

2. `02_mass_spring_cube.ipynb`
   - 8 masses
   - 28 springs
   - spring-damper force
   - ground contact
   - 12 actuators

3. `03_vectorized_2048_cubes.ipynb`
   - 2048 environments
   - 16,384 particle threads
   - vectorized observation/reward
   - throughput benchmark

4. `04_PPO_training.ipynb`
   - PyTorch actor-critic
   - GAE
   - PPO clipped objective
   - 2048-env GPU rollout
   - learned policy evaluation

## Colab
Runtime > Change runtime type > T4 GPU 권장.

모든 notebook은 첫 셀에서:
    pip install warp-lang==1.17.0
을 실행하도록 되어 있습니다.

GPU가 없으면 Warp CPU backend로 fallback하지만,
03/04의 2048-env 실습은 GPU 사용을 강력히 권장합니다.

## Teaching flow
- 01: 30–40분
- 02: 60–80분
- 03: 40–50분
- 04: 70–100분 + 과제

총 4–5시간 또는 2회차 수업에 적합합니다.