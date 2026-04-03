# CDT_simulation (통합 야드 시뮬레이션 환경)

본 워크스페이스는 기존에 분리되어 있던 **대규모 병렬 실험(Experiment)** 환경과 **3D 시각화(Visualization)** 환경을 하나의 코어 로직으로 시뮬레이션할 수 있도록 통합된 프로젝트입니다. 

이제 **하나의 적재 알고리즘을 작성하면, 눈으로 직접 검증(Visualization)하고 곧바로 대규모 성능 평가(Experiment)까지 원스톱으로 진행**할 수 있습니다.

---

## 1. 폴더 구조
```text
C:\workspace\CDT_simulation\
├── datas\
│   └── sim_data_251208_v2.csv       # 시뮬레이션 구동을 위한 원본 이벤트 데이터
├── simulation\
│   ├── config.py                    # 워크스페이스 공통 설정 (YardSimConfig)
│   ├── core.py                      # 핵심 로직 (YardSimulation, Strategy 패턴 클래스들)
│   └── visualization.py             # Panda3D 기반 3D 렌더링 엔진
├── run_visualization.py             # 단일 시나리오 3D 시각화 실행 스크립트
└── run_experiment.py                # 다중 시나리오 & 멀티프로세싱 대규모 실험 스크립트
```

---

## 2. 새로운 적재 로직(알고리즘) 추가하는 방법

적재 알고리즘을 테스트하려면 `BaseStackingStrategy`를 상속받는 새로운 전략 클래스를 `simulation/core.py` (또는 별도 분리된 파일)에 선언하기만 하면 됩니다.

### 예시: 내 맘대로 쌓는 전략 추가하기 (`simulation/core.py`에 추가)
```python
from simulation.core import BaseStackingStrategy
import random

class MyAwesomeStrategy(BaseStackingStrategy):
    def __init__(self, cdt_key: str):
        super().__init__(cdt_key)

    def find_position_for_inbound(self, sim, container_size, container_type, container_info):
        # 1. 반입(IN) 이벤트 발생 시 컨테이너를 어디에 둘지 정하는 로직
        # sim.gp_yard (일반 야드 3D 배열), sim.yard_depth (높이) 등 접근 가능
        # 조건에 맞는 (block, x, y, z, container_type) 튜플을 반환하면 됩니다.
        pass

    def find_position_for_rehandle(self, sim, container_size, container_type, container_info, exclude_position):
        # 2. 밑에 깔린 컨테이너를 빼내기 위해 이 컨테이너를 다시 옮겨야 할 때(Rehandling)
        # 어디로 옮길지 정하는 로직 (exclude_position: 원래 있던 자리는 제외하고 탐색)
        pass
```

---

## 3. 작성한 로직을 "시각화"로 눈으로 검증하기

`run_visualization.py` 파일의 전략 선택 부분을 방금 만든 클래스로 바꾼 뒤 실행합니다.

```python
# run_visualization.py 편집
from simulation.core import MyAwesomeStrategy 

# ... 
strategy = MyAwesomeStrategy(cdt_key=cfg.cdt_key)
hud_title = "Exp [My Awesome Logic]"
# ...
```

**[실행]** 터미널에 아래 명령어를 입력합니다.
```bash
cd C:\workspace\CDT_simulation
python run_visualization.py
```
* **조작법:** V키(시점변경), 마우스 드래그(회전), 휠(확대/축소), Space(일시정지/재생)

---

## 4. 검증된 로직을 "대규모 병렬 실험"으로 성능 평가하기

시각화를 통해 알고리즘에 버그가 없고 의도대로 쌓이는 것을 확인했다면, 이제 수십 가지 조건(블록 크기, 노이즈, 여러 번 반복 등)에 대해 백그라운드에서 고속으로 성능을 뽑아낼 차례입니다.

`run_experiment.py`의 `_execute_single_task` 함수 안에 전략을 추가해 줍니다.

```python
# run_experiment.py의 55번째 줄 부근
    elif strategy_name == 'MyAwesome':
        strategy = MyAwesomeStrategy(cdt_key=cdt_source)
```

**[실행]** 터미널에 아래 명령어로 병렬 연산을 시작합니다.
```bash
python run_experiment.py -w 8 -r 5
```
* `-w 8`: CPU 8코어를 사용하여 병렬 처리 (속도 8배 향상)
* `-r 5`: 각 실험 조건을 5번씩 반복(Repetition)하여 결과의 신뢰성 확보
* `-g 2 4 6`: GP 블록 크기를 2, 4, 6으로 다르게 설정하며 테스트 (선택사항)
* 완료되면 폴더에 `experiment_results.csv`에 모든 통계(평균 역전 비율, 재취급 횟수 등)가 깔끔하게 정리되어 출력됩니다.
