import pandas as pd
from simulation.core import YardSimulation, RandomStrategy, FloorByFloorStrategy, GlobalSearchStrategy
from simulation.config import YardSimConfig
from simulation.visualization import play_in_panda3d

# --- 1. 데이터 로드 ---
df = pd.read_csv("datas/sim_data_251208_v2.csv", parse_dates=["EVENT_TS"])

# --- 2. 기본 설정 ---
cfg = YardSimConfig(
    width=40,
    height=12,
    depth=5,
    gp_blocks=2,
    rf_blocks=2,
    cdt_key='CDT_true' # 분석할 기준 키 ('std_CDT_pred', 'no_std_CDT_pred', 'CDT_true' 등)
)

# --- 3. 실험(전략) 시나리오 선택 ---

# [1] Random
# strategy = RandomStrategy()
# hud_title = "Exp [Random]"

# [2] FloorByFloor (Smart Baseline)
strategy = FloorByFloorStrategy(cdt_key=cfg.cdt_key)
hud_title = f"Exp [FloorByFloor] - {cfg.cdt_key}"

# [3] GlobalSearch 
# strategy = GlobalSearchStrategy(cdt_key=cfg.cdt_key)
# hud_title = f"Exp [GlobalSearch] - {cfg.cdt_key}"

# ======================================================================

print(f"--- Running Visualization ---")
print(f"CFG: GP={cfg.gp_blocks}, RF={cfg.rf_blocks}, W={cfg.width}, H={cfg.height}, D={cfg.depth}")
print(f"STRATEGY: {strategy.__class__.__name__}")
print(f"CDT_KEY: {cfg.cdt_key}")

# --- 4. 시각화 엔진 평가 ---
# [NEW] 시각화 함수에 전략(Strategy) 클래스 기반 동작 방식을 주입합니다.
play_in_panda3d(
    df=df,
    yard_config=cfg,
    strategy_cls=strategy.__class__,
    ms_per_event=200,
    blocks_per_row=3,
    lane_every=None,
    colorful=True,
    window_title="Unified Yard Simulation",
    window_size=(1920, 1080),
    hud_title=hud_title,
    view_orientation="NE",
)
