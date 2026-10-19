# PushT Keyboard Simulator 실행 가이드 (Miniconda 설치 이후)

대상: `pusht_keyboard_sim.py`를 키보드/마우스 GUI로 실행하기 위한 **간단한 설정**입니다.

> 전제: Miniconda를 이미 설치했습니다. `pusht_keyboard_sim.py`는 로컬 PC에 있어야 합니다. 아래 명령은 **Anaconda Prompt(Windows)** 또는 **Terminal(macOS/Linux)** 에서 입력합니다.

## 1. 새 Conda 환경 만들기

```bash
conda create -n pusht python=3.11 -y
conda activate pusht
```

## 2. 필요한 패키지 설치

```bash
conda install numpy matplotlib tk -y
```

- `numpy`: 벡터·행렬·물리 계산
- `matplotlib`: 시뮬레이션 화면과 키보드 이벤트
- `tk`: Matplotlib의 대화형 GUI 백엔드(TkAgg)

> **참고:** 강의자료에 나오는 `pusht_imitation.py`까지 실행한다면 `torch` 등 추가 패키지가 필요할 수 있습니다. 이 가이드는 `pusht_keyboard_sim.py` 실행만을 위한 최소 설정입니다.

## 3. Python 파일이 있는 폴더로 이동

예시: 파일이 `Desktop/PushT` 폴더에 있는 경우

**Windows (Anaconda Prompt):**

```bat
cd /d "%USERPROFILE%\Desktop\PushT"
```

**macOS/Linux:**

```bash
cd ~/Desktop/PushT
```

`pusht_keyboard_sim.py`가 해당 폴더에 있는지 확인하세요.

## 4. 시뮬레이터 실행

```bash
python pusht_keyboard_sim.py
```

GUI 창이 열리면 강의자료의 안내에 따라 방향키/WASD 등으로 pusher의 목표 위치를 조작합니다. 키 배치는 실제 프로그램 화면의 도움말을 확인하세요.

## 5. 자주 생기는 문제

| 증상 | 해결 방법 |
|---|---|
| `conda` 명령을 찾을 수 없음 | Windows는 **Anaconda Prompt** 사용, macOS/Linux는 터미널을 다시 열거나 `conda init` 확인 |
| `ModuleNotFoundError: numpy` / `matplotlib` | `conda activate pusht` 후 `conda install numpy matplotlib -y` |
| `can't open file ...` | `cd`로 파일이 있는 폴더에 이동했는지, 파일명이 일치하는지 확인 |
| 창이 열리지 않음 / Tk 에러 | `conda install tk matplotlib -y` 후 GUI가 지원되는 로컬 터미널에서 다시 실행 |

## 다음 실행부터는

새 터미널에서 아래 3단계만 수행하면 됩니다.

```bash
conda activate pusht
cd 경로/PushT
python pusht_keyboard_sim.py
```

Windows에서는 `cd /d "C:\\경로\\PushT"` 형식으로 경로를 입력하세요.

---

**실행 흐름:** Miniconda 설치 → `pusht` 환경 생성 → 패키지 설치 → Python 파일 폴더 이동 → 실행
