# Tutorial: Teleoperation → 시연 데이터 → 모방학습

이 과제는 로봇 학습의 기본 파이프라인을 처음부터 끝까지 직접 해 보는 것입니다.
정책의 입력은 카메라 이미지가 아니라 **키포인트(위치) 정보**입니다: 로프 키포인트, 목표 키포인트, 막대 끝 위치.

```
 M0 설치 ──► M1 teleop ──► M2 시연 수집 · 검증 ──► M3 모방학습 구현 · 학습 ──► M4 평가 · 결과 영상 ──► M5 보고서 · PPT · YouTube
           (사람이 조종)   demos/<학번>/*.npz        il/*.py, ckpt.pt          eval.json, videos/*.mp4
```

각 단계 끝의 **✅ 확인** 항목이 그 milestone의 제출물입니다. 모든 명령은 저장소 최상위 폴더에서 실행합니다.

---

## M0. 설치와 실행

README의 [설치](README.md#설치)대로 가상환경을 만들고 패키지를 설치한 뒤 다음을 실행합니다.

```bash
python scripts/check_env.py --student <학번>
```

이 스크립트는 다음을 확인합니다.
- 장면 생성: UR5e 로봇, 막대, 로프 T, 테이블
- 관측: 85차원, 모두 유한한 값
- 제어: 막대를 50스텝 동안 내렸을 때 실제로 내려가는지
- 속도: 한 스텝이 100 ms(10 Hz)보다 빠른지

확인이 끝나면 위에서 내려다본 장면을 `runs/m0/scene.png`로 저장합니다. 그림에서 빨간 점은 로프, 초록 점은 목표입니다.

> **알아둘 것** — 시뮬레이터 코드를 열어 보세요. `sim/scene.py`의 `make_env(seed)`가 모든 스크립트의 출발점입니다.
> 반환되는 `env`는 `obs, reward, done, info = env.step(action)` 형태로 씁니다.
> `info['err']`는 목표까지의 위치 / 회전 / 모양 오차이고, `info['success']`는 성공 여부입니다.

**✅ 확인**: `runs/m0/check_env.json`에 `"passed": true`가 있고 `runs/m0/scene.png`가 있으면 됩니다.

---

## M1. Teleoperation

```bash
python teleop.py --student <학번>
```

1. 뷰어 창을 클릭해서 앞으로 가져옵니다. 키 입력은 뷰어 창이 앞에 있을 때만 의도대로 동작합니다.
2. 처음에는 막대 끝이 로프 위 3–4 cm에 있습니다. **PgDn**으로 막대를 테이블 가까이 내립니다. 로프 지름이 2 cm이므로 막대 끝 높이 5–10 mm가 적당합니다.
3. 방향키로 로프를 밀어서 빨간 키포인트를 초록 키포인트에 겹칩니다. 로프는 미는 대로 휘고 테이블 마찰 때문에 미끄러지기도 합니다.
   - 한 번에 크게 밀기보다 교차점 근처를 조금씩 미는 것이 쉽습니다.
   - 로프가 많이 돌아가 있으면 팔 끝을 밀어서 먼저 회전을 맞춥니다.
4. 터미널의 `pos / rot / shape`가 모두 기준 안에 들어오면 `SUCCESS`가 뜹니다. 그때 **Enter**로 저장합니다.
   - 망쳤으면 **Delete**로 같은 시드를 다시 시작합니다.
5. 60초(600스텝)가 지나면 에피소드가 자동으로 끝나고 저장됩니다.

저장된 파일: `demos/<학번>/ep_<seed>.npz`. 시드는 학번 끝 4자리 × 100부터 차례로 씁니다. 다시 실행하면 이미 저장한 시드는 건너뜁니다.

화면 녹화(30초 이상, 성공 순간 포함)를 `runs/m1/teleop.mp4`로 저장합니다.
- Windows: Win+Alt+R (Xbox Game Bar)
- macOS: Cmd+Shift+5
- Linux: OBS 등

**✅ 확인**: 성공한 에피소드(`success=True`)가 1개 이상 있고, `runs/m1/teleop.mp4`가 있으면 됩니다.

---

## M2. 시연 데이터 수집과 검증

모방학습은 데이터의 양과 질이 성능을 거의 결정합니다. **30개 이상**의 에피소드를 모읍니다.

### 데이터 형식

`sim/episode_io.py` 맨 위에 모든 키가 설명되어 있습니다. 핵심만 보면 다음과 같습니다.

| 키 | 모양 | 의미 |
|---|---|---|
| `obs` | (T, 85) | 스텝 k에서 **행동하기 전**의 관측 |
| `act` | (T, 3) | 스텝 k에서 실제로 실행된 행동 |
| `human` | (T,) | 그 스텝에 키를 누르고 있었는지 |
| `kp`, `tip`, `goal` | | 로프 키포인트, 막대 끝, 목표 (테이블 좌표, m) |
| `err`, `succ` | (T, 3), (T,) | 행동 후 오차와 성공 여부 |

즉 학습 쌍은 `(obs[k], act[k])`입니다.

`human=False`인 스텝은 조종자가 생각하느라 멈춘 시간입니다. 이 스텝을 그대로 학습하면 정책이 "가만히 있기"를 배웁니다. 그래서 `il/dataset.py`의 `load_demos`는 기본으로 이 스텝을 뺍니다.

### 검증

```bash
python scripts/inspect_demos.py demos/<학번>          # 형식 검사, 에피소드별 요약, 그림 → runs/m2/
python scripts/replay_demo.py demos/<학번>/ep_XXXX.npz         # 저장한 행동을 다시 시뮬레이션
python scripts/replay_demo.py demos/<학번>/ep_XXXX.npz --view  # 뷰어로 재생
```

`replay_demo.py`는 같은 시드에서 저장된 행동만 다시 넣었을 때 로프가 똑같이 움직이는지 봅니다.
같은 컴퓨터라면 drift가 0.0 mm여야 합니다. 이것으로 기록이 관측과 행동을 한 스텝도 어긋나지 않게 저장했는지 확인할 수 있습니다.

> **생각해 볼 것** (보고서에 씀)
> - 시연이 얼마나 일관적인가요? 같은 상황에서 매번 비슷하게 밀었나요?
> - 키보드 행동은 8방향 × 2속도의 이산값입니다. 이것이 회귀 모델 학습에 어떤 영향을 줄까요?
> - 실패한 시연도 학습에 넣어야 할까요? (`--only-success`로 비교해 볼 수 있습니다.)

**✅ 확인**: 다음이 모두 있으면 됩니다.
- `runs/m2/summary.json`: valid ≥ 30, 모든 시드가 본인 시드 범위 안
- `runs/m2/summary.csv`
- `runs/m2/demos.png`

---

## M3. 모방학습 구현 (Behaviour Cloning)

목표는 정책 π(행동 | 위치 관측)를 지도학습으로 배우는 것입니다. `il/`의 `TODO`를 채웁니다.

| 파일 | 채울 것 |
|---|---|
| `il/dataset.py` | `make_features`: 85차원 관측에서 위치 특징을 고릅니다 |
| | `make_pairs`: 관측 창과 행동 창을 만듭니다. 에피소드 경계를 넘지 않게 합니다 |
| | `Normalizer`, `BCDataset` |
| `il/policy.py` | `MLPPolicy`: 신경망 |
| | `Agent.act`: 시뮬레이터 관측 1개를 받아 행동 1개를 냅니다 |
| `il/train_bc.py` | 학습 루프: train / val 손실, best checkpoint 저장 |

순서대로 하면 됩니다.

```bash
python il/check_il.py --data demos/<학번>                         # 1. 구현 검사 (학습 없음). 6개 모두 PASS
python il/train_bc.py --data demos/<학번> --out runs/m3 --dry-run # 2. 배치 하나로 forward/backward
python il/train_bc.py --data demos/<학번> --out runs/m3           # 3. 학습 → ckpt.pt, train_log.csv
```

설계할 때 고려할 점
- **특징**: 로프와 목표의 차이, 막대와 로프의 상대 위치처럼 과제에 맞는 표현이 절대 좌표보다 일반화가 잘 될 수 있습니다.
- **분할**: train / val은 에피소드 단위로 나눕니다. `split_by_episode`가 이미 그렇게 합니다. 이유는 보고서에 씁니다.
- **정규화**: 관측과 행동 모두 train 데이터로만 평균과 표준편차를 구합니다.
- **관측·행동 창**: `--obs-horizon`, `--act-horizon`. 행동 묶음(action chunking)은 선택 사항입니다. 키보드 시연처럼 멈칫거리는 데이터에서는 도움이 될 수 있습니다.

**✅ 확인**: 다음이 모두 있으면 됩니다.
- `runs/m3/check_il.json`: `"passed": true`
- `runs/m3/ckpt.pt`, `train_log.csv`, `config.json`
- 보고서에 손실 곡선

---

## M4. 폐루프 평가

학습 손실이 낮다고 정책이 잘 동작하는 것은 아닙니다. 정책이 낸 행동이 다음 관측을 바꾸기 때문에, 작은 오차가 쌓여 시연에서 본 적 없는 상태로 가게 됩니다(compounding error / covariate shift).
그래서 반드시 시뮬레이터에서 직접 돌려 봐야 합니다.

```bash
python il/eval_policy.py --agent zero                              # 기준선: 가만히 있기
python il/eval_policy.py --agent bc --ckpt runs/m3/ckpt.pt         # 내 정책, 평가 시드 50개
python il/eval_policy.py --agent bc --ckpt runs/m3/ckpt.pt --n 3 --view   # 눈으로 보기
```

- 평가 시드(900000–900049)는 모든 학생이 같고, 시연에는 쓰지 않은 시작 자세입니다.
- 성공은 **마지막 스텝**에서 판정합니다. 한 번 맞췄다가 다시 밀어서 흐트러뜨리면 실패입니다. `ever`는 참고용입니다.
- 결과 파일:
  - `runs/m4/<agent>/eval.json`
  - `runs/m4/<agent>/rollouts/ep_<seed>.npz`: 시연과 같은 형식이라 `scripts/inspect_demos.py runs/m4/bc/rollouts`로 그림을 그릴 수 있습니다.

### 결과 영상 만들기

평가가 끝나면 rollout을 mp4로 만듭니다. 저장된 MuJoCo 상태로 렌더링하므로 평가 결과와 똑같은 장면이 나옵니다.
영상은 H.264라서 PowerPoint에 바로 넣을 수 있고 YouTube에도 그대로 올릴 수 있습니다.
- 로봇 팔은 반투명으로 그립니다.
- 빨간 점은 로프, 초록 점은 목표입니다.
- 화면 위쪽에 시간, 오차, `SUCCESS`가 표시됩니다.

```bash
# eval.json에서 성공한 시드와 실패한 시드를 골라서 (성공이 있으면 성공 1개 이상 + 실패 1개 이상, 모두 3개 이상)
python scripts/render_video.py runs/m4/bc/rollouts/ep_900003.npz runs/m4/bc/rollouts/ep_900010.npz runs/m4/bc/rollouts/ep_900021.npz
# → runs/m4/bc/videos/ep_900003.mp4 ...   (--speed 2: 두 배속)
```

시연도 같은 방법으로 영상을 만들 수 있습니다. 발표에서 teleop과 정책을 비교할 때 씁니다.

```bash
python scripts/render_video.py demos/<학번>/ep_XXXX.npz --out runs/m1/videos
```

**✅ 확인**: 다음이 모두 있으면 됩니다.
- `runs/m4/zero/eval.json`과 `runs/m4/bc/eval.json`: 둘 다 `n = 50`
- `runs/m4/bc/videos/`: 결과 영상 3개 이상 (성공·실패 포함)

---

## M5. 최종 보고서, 발표 PPT, YouTube

**1. 보고서** `report.pdf`

[docs/report_template.md](docs/report_template.md) 양식으로 씁니다. 다음 내용을 담습니다.
- 데이터 분석
- 설계 선택과 그 이유
- 손실 곡선
- 성공률 (기준선과 비교)
- 결과 영상을 근거로 한 실패 사례 분석
- ablation 1개 이상 (예: 시연 개수, 특징, `--only-success`, horizon)
- YouTube 링크

**2. 발표 PPT** `slides.pptx` (또는 `slides.pdf`)

10장 내외로 다음 흐름을 따릅니다.
1. 문제 정의
2. teleop과 데이터
3. 모델 설계 (키포인트 특징)
4. 결과: 성공률 표, 결과 영상
5. 실패 분석
6. 결론

결과 영상은 `runs/m4/bc/videos/*.mp4`를 슬라이드에 직접 넣습니다(삽입 → 비디오).

**3. YouTube 발표 영상** `youtube_url.txt`

- 5–10분 동안 PPT로 설명하면서 다음 장면을 보여 줍니다.
  - teleop 장면 (M1 녹화)
  - 결과 영상 (성공·실패)
- 목소리 대신 자막으로 설명해도 됩니다.
- YouTube에 **일부 공개(unlisted)** 이상으로 올립니다.
- 링크 한 줄만 `youtube_url.txt`에 저장합니다. 예: `https://youtu.be/xxxxxxxxxxx`
- 영상 속 결과는 제출한 `eval.json`, 영상 파일과 같아야 합니다.

## 제출

```bash
python scripts/make_submission.py --student <학번>    # 빠진 파일을 알려줍니다
```

`submission_<학번>.zip`을 제출합니다. zip에는 보고서, PPT, 결과 영상, YouTube 링크가 모두 들어갑니다.
