import numpy as np
import pandas as pd
from typing import Optional
import random

from .config import YardSimConfig

# ======================================================================
# Strategy Pattern Base
# ======================================================================

class BaseStackingStrategy:
    """
    적재 전략을 위한 기본(추상) 클래스입니다.
    이 전략들은 YardSimulation 객체의 규칙(find_position_for_inbound 등)을
    주입받아 동작합니다.
    """
    def __init__(self, cdt_key: str = None):
        self.cdt_key = cdt_key

    def find_position_for_inbound(self, sim: 'YardSimulation', container_size, container_type, container_info):
        """[신규 반입] 컨테이너를 위한 위치 탐색"""
        raise NotImplementedError("Subclasses must implement find_position_for_inbound method.")

    def find_position_for_rehandle(self, sim: 'YardSimulation', container_size, container_type, container_info, exclude_position):
        """[리핸들링] 컨테이너를 위한 위치 탐색"""
        raise NotImplementedError("Subclasses must implement find_position_for_rehandle method.")


# ======================================================================
# Specific Strategies
# ======================================================================

class RandomStrategy(BaseStackingStrategy):
    """
    [전략 1] 층별 무작위(랜덤) 적재 전략
    * 1층을 무작위로 채우고, 자리가 없으면 2층으로 넘어갑니다.
    """
    def __init__(self):
        super().__init__(cdt_key=None)

    def find_position_for_inbound(self, sim: 'YardSimulation', container_size, container_type, container_info):
        yard = sim.rf_yard if container_type == 'RF' else sim.gp_yard
        blocks_count = sim.rf_blocks if container_type == 'RF' else sim.gp_blocks
        
        for z in range(sim.yard_depth):
            available_on_this_floor = []

            for block in range(blocks_count):
                if container_size == '20ft':
                    for x in range(0, sim.split_x): 
                        for y in range(sim.yard_height):
                            if yard[block, x, y, z] is None:
                                if z == 0 or yard[block, x, y, z-1] is not None:
                                    available_on_this_floor.append((block, x, y, z, container_type))
                                    
                elif container_size == '40ft':
                    for x in range(sim.split_x, sim.yard_width, 2): 
                        if x + 1 < sim.yard_width:
                            for y in range(sim.yard_height):
                                if yard[block, x, y, z] is None and yard[block, x+1, y, z] is None:
                                    if z == 0 or (yard[block, x, y, z-1] is not None and yard[block, x+1, y, z-1] is not None):
                                        available_on_this_floor.append((block, x, y, z, container_type))
            
            if available_on_this_floor:
                return random.choice(available_on_this_floor)
                
        return None

    def find_position_for_rehandle(self, sim: 'YardSimulation', container_size, container_type, container_info, exclude_position):
        yard = sim.rf_yard if container_type == 'RF' else sim.gp_yard
        source_block = exclude_position[0]
        exclude_pos_4d = exclude_position[:4]
        
        for z in range(sim.yard_depth):
            available_on_this_floor = []

            if container_size == '20ft':
                for x in range(0, sim.split_x): 
                    for y in range(sim.yard_height):
                        pos_check = (source_block, x, y, z)
                        if pos_check == exclude_pos_4d: continue 
                        if yard[source_block, x, y, z] is None:
                            if z == 0 or yard[source_block, x, y, z-1] is not None:
                                available_on_this_floor.append((source_block, x, y, z, container_type))
                                
            elif container_size == '40ft':
                for x in range(sim.split_x, sim.yard_width, 2): 
                    if x + 1 < sim.yard_width:
                        for y in range(sim.yard_height):
                            pos_check = (source_block, x, y, z)
                            if pos_check == exclude_pos_4d: continue 
                            if yard[source_block, x, y, z] is None and yard[source_block, x+1, y, z] is None:
                                if z == 0 or (yard[source_block, x, y, z-1] is not None and yard[source_block, x+1, y, z-1] is not None):
                                    available_on_this_floor.append((source_block, x, y, z, container_type))
            
            if available_on_this_floor:
                return random.choice(available_on_this_floor)
                
        return None

class FloorByFloorStrategy(BaseStackingStrategy):
    """
    [전략 2] 층별(Floor-by-Floor) 최소 CDT 차이 적재 전략 (Smart)
    """
    def __init__(self, cdt_key: str):
        super().__init__(cdt_key)

    def find_position_for_inbound(self, sim: 'YardSimulation', container_size, container_type, container_info):
        new_cdt = container_info.get(self.cdt_key, 0)
        yard = sim.rf_yard if container_type == 'RF' else sim.gp_yard
        blocks_count = sim.rf_blocks if container_type == 'RF' else sim.gp_blocks

        # 1. 1층(z=0)은 무작위
        floor_zero_positions = []
        z = 0  
        for block in range(blocks_count):
            if container_size == '20ft':
                for x in range(0, sim.split_x): 
                    for y in range(sim.yard_height):
                        if yard[block, x, y, z] is None:
                            floor_zero_positions.append((block, x, y, z, container_type))
            elif container_size == '40ft':
                for x in range(sim.split_x, sim.yard_width, 2): 
                    if x + 1 < sim.yard_width:
                        for y in range(sim.yard_height): 
                            if yard[block, x, y, z] is None and yard[block, x+1, y, z] is None:
                                floor_zero_positions.append((block, x, y, z, container_type))
        if floor_zero_positions:
            return random.choice(floor_zero_positions)

        # 2. 2층(z>0)부터 같은 층 내에서 최소 CDT 탐색
        for z in range(1, sim.yard_depth):
            best_position_on_this_floor = None
            min_cdt_diff_on_this_floor = float('inf') 

            for block in range(blocks_count):
                if container_size == '20ft':
                    for x in range(0, sim.split_x): 
                        for y in range(sim.yard_height):
                            if yard[block, x, y, z] is None:
                                below_key = yard[block, x, y, z-1]
                                if below_key:
                                    real_key = str(below_key).replace('_2nd', '')
                                    if real_key in sim.containers:
                                        below_cdt = sim.containers[real_key].get(self.cdt_key, 0)
                                        cdt_diff = new_cdt - below_cdt
                                        if cdt_diff < min_cdt_diff_on_this_floor:
                                            min_cdt_diff_on_this_floor = cdt_diff
                                            best_position_on_this_floor = (block, x, y, z, container_type)

                elif container_size == '40ft':
                    for x in range(sim.split_x, sim.yard_width, 2): 
                        if x + 1 < sim.yard_width:
                            for y in range(sim.yard_height): 
                                if yard[block, x, y, z] is None and yard[block, x+1, y, z] is None:
                                    key_left = yard[block, x, y, z-1]
                                    key_right = yard[block, x+1, y, z-1]
                                    if key_left is not None and key_right is not None:
                                        supporters = set()
                                        supporters.add(str(key_left).replace('_2nd', ''))
                                        supporters.add(str(key_right).replace('_2nd', ''))
                                        
                                        current_cdts = [sim.containers[k].get(self.cdt_key, 0) for k in supporters if k in sim.containers]
                                        
                                        if len(current_cdts) == len(supporters):
                                            min_supporter_cdt = min(current_cdts)
                                            cdt_diff = new_cdt - min_supporter_cdt
                                            if cdt_diff < min_cdt_diff_on_this_floor:
                                                min_cdt_diff_on_this_floor = cdt_diff
                                                best_position_on_this_floor = (block, x, y, z, container_type)
            if best_position_on_this_floor:
                return best_position_on_this_floor
        return None

    def find_position_for_rehandle(self, sim: 'YardSimulation', container_size, container_type, container_info, exclude_position):
        new_cdt = container_info.get(self.cdt_key, 0)
        yard = sim.rf_yard if container_type == 'RF' else sim.gp_yard
        source_block = exclude_position[0]
        exclude_pos_4d = exclude_position[:4]

        # 1층(z=0) 랜덤 (Source Block 내부)
        z = 0
        available_floor_zero = []
        if container_size == '20ft':
            for x in range(0, sim.split_x):
                for y in range(sim.yard_height):
                    pos_check = (source_block, x, y, z)
                    if pos_check != exclude_pos_4d and yard[source_block, x, y, z] is None:
                        available_floor_zero.append((source_block, x, y, z, container_type))
        elif container_size == '40ft':
             for x in range(sim.split_x, sim.yard_width, 2):
                if x + 1 < sim.yard_width:
                    for y in range(sim.yard_height):
                        pos_check = (source_block, x, y, z)
                        if pos_check != exclude_pos_4d and yard[source_block, x, y, z] is None and yard[source_block, x+1, y, z] is None:
                             available_floor_zero.append((source_block, x, y, z, container_type))
        if available_floor_zero:
            return random.choice(available_floor_zero)

        # 2. 2층(z>0) 이상 같은 층 내 최소 CDT 탐색
        for z in range(1, sim.yard_depth):
            best_position_on_this_floor = None
            min_cdt_diff_on_this_floor = float('inf')
            
            if container_size == '20ft':
                for x in range(0, sim.split_x):
                    for y in range(sim.yard_height):
                        pos_check = (source_block, x, y, z)
                        if pos_check == exclude_pos_4d: continue
                        if yard[source_block, x, y, z] is None:
                            below_key = yard[source_block, x, y, z-1]
                            if below_key:
                                real_key = str(below_key).replace('_2nd', '')
                                if real_key in sim.containers:
                                    below_cdt = sim.containers[real_key].get(self.cdt_key, 0)
                                    cdt_diff = new_cdt - below_cdt
                                    if cdt_diff < min_cdt_diff_on_this_floor:
                                        min_cdt_diff_on_this_floor = cdt_diff
                                        best_position_on_this_floor = (source_block, x, y, z, container_type)

            elif container_size == '40ft':
                for x in range(sim.split_x, sim.yard_width, 2):
                    if x + 1 < sim.yard_width:
                        for y in range(sim.yard_height):
                            pos_check = (source_block, x, y, z)
                            if pos_check == exclude_pos_4d: continue
                            if yard[source_block, x, y, z] is None and yard[source_block, x+1, y, z] is None:
                                key_left = yard[source_block, x, y, z-1]
                                key_right = yard[source_block, x+1, y, z-1]
                                if key_left is not None and key_right is not None:
                                    supporters = set()
                                    supporters.add(str(key_left).replace('_2nd', ''))
                                    supporters.add(str(key_right).replace('_2nd', ''))
                                    current_cdts = [sim.containers[k].get(self.cdt_key, 0) for k in supporters if k in sim.containers]
                                    if len(current_cdts) == len(supporters):
                                        min_supporter_cdt = min(current_cdts)
                                        cdt_diff = new_cdt - min_supporter_cdt
                                        if cdt_diff < min_cdt_diff_on_this_floor:
                                            min_cdt_diff_on_this_floor = cdt_diff
                                            best_position_on_this_floor = (source_block, x, y, z, container_type)

            if best_position_on_this_floor:
                return best_position_on_this_floor
        return None

class GlobalSearchStrategy(BaseStackingStrategy):
    """
    [전략 3] 전역(Global Search) 최소 CDT 차이 적재 전략
    (층 구애받지 않고 가장 효율적인 빈 자리 탐색)
    """
    def __init__(self, cdt_key: str):
        super().__init__(cdt_key)

    def find_position_for_inbound(self, sim: 'YardSimulation', container_size, container_type, container_info):
        new_cdt = container_info.get(self.cdt_key, 0)
        yard = sim.rf_yard if container_type == 'RF' else sim.gp_yard
        blocks_count = sim.rf_blocks if container_type == 'RF' else sim.gp_blocks

        # 1층 무작위
        floor_zero_positions = []
        z = 0 
        for block in range(blocks_count):
            if container_size == '20ft':
                for x in range(0, sim.split_x): 
                    for y in range(sim.yard_height):
                        if yard[block, x, y, z] is None:
                            floor_zero_positions.append((block, x, y, z, container_type))
            elif container_size == '40ft':
                for x in range(sim.split_x, sim.yard_width, 2): 
                    if x + 1 < sim.yard_width:
                        for y in range(sim.yard_height): 
                            if yard[block, x, y, z] is None and yard[block, x+1, y, z] is None:
                                floor_zero_positions.append((block, x, y, z, container_type))
        if floor_zero_positions:
            return random.choice(floor_zero_positions)

        # 2층 이상 전체 탐색
        best_position_global = None
        min_cdt_diff_global = float('inf') 

        for z in range(1, sim.yard_depth):
            for block in range(blocks_count):
                if container_size == '20ft':
                    for x in range(0, sim.split_x): 
                        for y in range(sim.yard_height):
                            if yard[block, x, y, z] is None:
                                below_key = yard[block, x, y, z-1]
                                if below_key:
                                    real_key = str(below_key).replace('_2nd', '')
                                    if real_key in sim.containers:
                                        below_cdt = sim.containers[real_key].get(self.cdt_key, 0)
                                        cdt_diff = new_cdt - below_cdt
                                        if cdt_diff < min_cdt_diff_global:
                                            min_cdt_diff_global = cdt_diff
                                            best_position_global = (block, x, y, z, container_type)

                elif container_size == '40ft':
                    for x in range(sim.split_x, sim.yard_width, 2): 
                        if x + 1 < sim.yard_width:
                            for y in range(sim.yard_height): 
                                if yard[block, x, y, z] is None and yard[block, x+1, y, z] is None:
                                    key_left = yard[block, x, y, z-1]
                                    key_right = yard[block, x+1, y, z-1]
                                    if key_left is not None and key_right is not None:
                                        supporters = set()
                                        supporters.add(str(key_left).replace('_2nd', ''))
                                        supporters.add(str(key_right).replace('_2nd', ''))
                                        current_cdts = [sim.containers[k].get(self.cdt_key, 0) for k in supporters if k in sim.containers]
                                        if len(current_cdts) == len(supporters):
                                            min_supporter_cdt = min(current_cdts)
                                            cdt_diff = new_cdt - min_supporter_cdt
                                            if cdt_diff < min_cdt_diff_global:
                                                min_cdt_diff_global = cdt_diff
                                                best_position_global = (block, x, y, z, container_type)
        return best_position_global

    def find_position_for_rehandle(self, sim: 'YardSimulation', container_size, container_type, container_info, exclude_position):
        new_cdt = container_info.get(self.cdt_key, 0)
        yard = sim.rf_yard if container_type == 'RF' else sim.gp_yard
        source_block = exclude_position[0] 
        exclude_pos_4d = exclude_position[:4]

        # 1층 무작위
        z = 0
        available_floor_zero = []
        if container_size == '20ft':
            for x in range(0, sim.split_x):
                for y in range(sim.yard_height):
                    pos_check = (source_block, x, y, z)
                    if pos_check != exclude_pos_4d and yard[source_block, x, y, z] is None:
                        available_floor_zero.append((source_block, x, y, z, container_type))
        elif container_size == '40ft':
             for x in range(sim.split_x, sim.yard_width, 2):
                if x + 1 < sim.yard_width:
                    for y in range(sim.yard_height):
                        pos_check = (source_block, x, y, z)
                        if pos_check != exclude_pos_4d and yard[source_block, x, y, z] is None and yard[source_block, x+1, y, z] is None:
                             available_floor_zero.append((source_block, x, y, z, container_type))
        if available_floor_zero:
            return random.choice(available_floor_zero)

        # 2층 이상 Source Block 내 전역 탐색
        best_position_global = None
        min_cdt_diff_global = float('inf')

        for z in range(1, sim.yard_depth):
            if container_size == '20ft':
                for x in range(0, sim.split_x):
                    for y in range(sim.yard_height):
                        pos_check = (source_block, x, y, z)
                        if pos_check == exclude_pos_4d: continue
                        if yard[source_block, x, y, z] is None:
                            below_key = yard[source_block, x, y, z-1]
                            if below_key:
                                real_key = str(below_key).replace('_2nd', '')
                                if real_key in sim.containers:
                                    below_cdt = sim.containers[real_key].get(self.cdt_key, 0)
                                    cdt_diff = new_cdt - below_cdt
                                    if cdt_diff < min_cdt_diff_global:
                                        min_cdt_diff_global = cdt_diff
                                        best_position_global = (source_block, x, y, z, container_type)

            elif container_size == '40ft':
                for x in range(sim.split_x, sim.yard_width, 2):
                    if x + 1 < sim.yard_width:
                        for y in range(sim.yard_height):
                            pos_check = (source_block, x, y, z)
                            if pos_check == exclude_pos_4d: continue
                            if yard[source_block, x, y, z] is None and yard[source_block, x+1, y, z] is None:
                                key_left = yard[source_block, x, y, z-1]
                                key_right = yard[source_block, x+1, y, z-1]
                                if key_left is not None and key_right is not None:
                                    supporters = set()
                                    supporters.add(str(key_left).replace('_2nd', ''))
                                    supporters.add(str(key_right).replace('_2nd', ''))
                                    current_cdts = [sim.containers[k].get(self.cdt_key, 0) for k in supporters if k in sim.containers]
                                    if len(current_cdts) == len(supporters):
                                        min_supporter_cdt = min(current_cdts)
                                        cdt_diff = new_cdt - min_supporter_cdt
                                        if cdt_diff < min_cdt_diff_global:
                                            min_cdt_diff_global = cdt_diff
                                            best_position_global = (source_block, x, y, z, container_type)

        return best_position_global

# ======================================================================
# Main Simulation Orchestrator
# ======================================================================

class YardSimulation:
    """
    통합된 Yard Simulation 클래스.
    생성자에서 주입받은 전략(`stacking_strategy`)을 통해 위치를 계산합니다.
    """
    def __init__(self, 
                 yard_config: Optional[YardSimConfig] = None,
                 stacking_strategy: BaseStackingStrategy = None):
        
        self.yard_config = yard_config or YardSimConfig()
        self.yard_width = self.yard_config.width
        self.yard_height = self.yard_config.height
        self.yard_depth = self.yard_config.depth
        
        self.gp_blocks = self.yard_config.gp_blocks
        self.rf_blocks = self.yard_config.rf_blocks
        
        self.split_x = int(self.yard_width * 0.3)
        if self.split_x % 2 != 0:
            self.split_x -= 1
        
        self.gp_yard = np.full((self.gp_blocks, self.yard_width, self.yard_height, self.yard_depth), None, dtype=object)
        self.rf_yard = np.full((self.rf_blocks, self.yard_width, self.yard_height, self.yard_depth), None, dtype=object)

        self.containers = {}
        self.container_positions = {}
        self.current_time = None
        self.event_history = []
        self.rehandling_count = 0

        self.temp_yard_containers = set()  
        self.overflow_count = 0            

        self.gp_total_capacity = self.gp_blocks * self.yard_width * self.yard_height * self.yard_depth
        if self.gp_total_capacity == 0: self.gp_total_capacity = 1 
        
        self.rf_total_capacity = self.rf_blocks * self.yard_width * self.yard_height * self.yard_depth
        if self.rf_total_capacity == 0: self.rf_total_capacity = 1 

        self.ignore_updates = self.yard_config.ignore_updates
        
        # 전략 주입 (DI)
        if stacking_strategy is None:
            self.stacking_strategy = RandomStrategy()
        else:
            self.stacking_strategy = stacking_strategy

    def find_available_position(self, container_size, container_type, container_info):
        return self.stacking_strategy.find_position_for_inbound(
            self, container_size, container_type, container_info
        )

    def find_position_for_rehandling(self, container_size, container_type, container_info, exclude_position):
        return self.stacking_strategy.find_position_for_rehandle(
            self, container_size, container_type, container_info, exclude_position
        )

    def run(self, df):
        """실험 코드를 위한 전체 실행 모드 (시각화 모드는 개별 이벤트별로 process_event를 호출)"""
        if self.ignore_updates:
            df_processed = df[df['EVENT_TYPE'].isin(['IN', 'OUT'])].copy()
        else:
            df_processed = df.copy()

        if df_processed.empty:
            return {}, 0, {}
            
        df_sorted = df_processed.sort_values('EVENT_TS')
        self.current_time = pd.to_datetime(df_sorted.iloc[0]['EVENT_TS'])

        simulation_stopped_early = False
        inversion_rates = []

        for idx, event in df_sorted.iterrows():
            should_continue = self.process_event(event)
            
            # 매 이벤트 직후 역전 비율 계산 기록
            current_inv_rate = self.calculate_current_inversion_rate()
            inversion_rates.append(current_inv_rate)

            if not should_continue:
                simulation_stopped_early = True
                break
        
        final_status = self.get_yard_status()
        avg_inversion_rate = np.mean(inversion_rates) if inversion_rates else 0.0

        if not self.event_history:
            return final_status, self.rehandling_count, {}

        df_history = pd.DataFrame(self.event_history)
        df_history['gp_storage_rate'] = df_history.get('gp_occupancy_slots', 0).fillna(0) / self.gp_total_capacity
        
        gp_avg_rate = df_history['gp_storage_rate'].mean()
        gp_median_rate = df_history['gp_storage_rate'].median()
        gp_max_rate = df_history['gp_storage_rate'].max()
        rf_max_rate = df_history['rf_occupancy_slots'].max() / self.rf_total_capacity if 'rf_occupancy_slots' in df_history else 0

        storage_stats = {
            'gp_avg': gp_avg_rate,
            'gp_median': gp_median_rate,
            'gp_max': gp_max_rate,
            'rf_max': rf_max_rate,
            'stopped_early': simulation_stopped_early, 
            'overflow_count': self.overflow_count,
            'avg_inversion_rate': avg_inversion_rate
        }
        
        return final_status, self.rehandling_count, storage_stats

    def calculate_current_inversion_rate(self):
        """현재 야드의 역전(Inverted) 컨테이너 비율. 늦게 나가는 화물(CDT 큼)이 빨리 나가는 화물(CDT 작음) 위에 있는 경우."""
        total_containers = len(self.containers)
        if total_containers == 0:
            return 0.0

        inverted_count = 0
        measure_key = 'CDT_true' # 기준은 항상 객관적인 CDT_true를 사용

        for yard_arr, blocks in [(self.gp_yard, self.gp_blocks), (self.rf_yard, self.rf_blocks)]:
            for z in range(1, self.yard_depth): # 밑에 깔린게 있는 블록만
                for b in range(blocks):
                    for y in range(self.yard_height):
                        for x in range(self.yard_width):
                            current_key = yard_arr[b, x, y, z]
                            if current_key is None: continue
                            if isinstance(current_key, str) and current_key.endswith('_2nd'): continue

                            if current_key not in self.containers: continue
                            my_info = self.containers[current_key]
                            my_cdt = my_info.get(measure_key, 0)
                            
                            supporters_cdts = []
                            container_size = self.get_container_size(my_info.get('CARGO_SIZE'))
                            
                            # (1) 내 바로 아래
                            below_key = yard_arr[b, x, y, z-1]
                            if below_key:
                                real_below = str(below_key).replace('_2nd', '')
                                if real_below in self.containers:
                                    supporters_cdts.append(self.containers[real_below].get(measure_key, 0))
                            
                            # (2) 40ft인 경우 오른쪽도
                            if container_size == '40ft' and x + 1 < self.yard_width:
                                below_right = yard_arr[b, x+1, y, z-1]
                                if below_right:
                                    real_right = str(below_right).replace('_2nd', '')
                                    if real_right in self.containers:
                                        supporters_cdts.append(self.containers[real_right].get(measure_key, 0))
                            
                            if supporters_cdts:
                                min_support_cdt = min(supporters_cdts)
                                if float(min_support_cdt) < float(my_cdt):
                                    inverted_count += 1
                                    
        return inverted_count / total_containers

    def process_event(self, event):
        event_time = pd.to_datetime(event['EVENT_TS'])
        unique_key = event['UNIQUE_KEY']
        event_type = event['EVENT_TYPE']

        if self.current_time is not None and event_time > self.current_time:
            time_diff = (event_time - self.current_time).total_seconds() / 3600
            self.update_container_times(time_diff)
            self.current_time = event_time
        else:
            self.current_time = event_time

        if event_type == 'IN':
            cargo_size = event.get('CARGO_SIZE', event.get('SZTP2'))
            cargo_type = event.get('CARGO_TYPE')
            if pd.isna(cargo_type) and 'SZTP2' in event:
                sztp = str(event['SZTP2'])
                if 'R' in sztp: cargo_type = 'RF'
                else: cargo_type = 'GP'
            
            container_info = {
                'CARGO_SIZE': cargo_size,
                'CARGO_TYPE': cargo_type,
                'std_CDT_pred': event.get('std_CDT_pred', 0),
                'no_std_CDT_pred': event.get('no_std_CDT_pred', 0),
                'CDT_true': event.get('CDT_true', 0),
                'consignee_name': event.get('consignee_name'), 
                'raw': event.get('raw'),
                'event_history': [event_type],
                'EVENT_TS': event_time
            }
            container_size = self.get_container_size(cargo_size)
            container_type = self.get_container_type(cargo_type)

            position = self.find_available_position(container_size, container_type, container_info)

            if position:
                self.place_container(unique_key, container_info, position)
            else:
                self.temp_yard_containers.add(unique_key)
                self.overflow_count += 1
                return False

        elif event_type in ['CUS', 'COR', 'COP']:
            if not self.ignore_updates:
                if unique_key in self.containers:
                    self.containers[unique_key]['std_CDT_pred'] = event.get('std_CDT_pred', self.containers[unique_key].get('std_CDT_pred', 0))
                    self.containers[unique_key]['no_std_CDT_pred'] = event.get('no_std_CDT_pred', self.containers[unique_key].get('no_std_CDT_pred', 0))
                    self.containers[unique_key]['CDT_true'] = event.get('CDT_true', self.containers[unique_key].get('CDT_true', 0))
                    self.containers[unique_key]['event_history'].append(event_type)

        elif event_type == 'OUT':
            if unique_key in self.containers:
                success = self.remove_container(unique_key)
                if not success:
                    return False
            elif unique_key in self.temp_yard_containers:
                self.temp_yard_containers.remove(unique_key)

        gp_occupied = np.count_nonzero(self.gp_yard != None)
        rf_occupied = np.count_nonzero(self.rf_yard != None)
        
        self.event_history.append({
            'timestamp': self.current_time,
            'unique_key': unique_key,
            'event_type': event_type,
            'gp_occupancy_slots': gp_occupied,
            'rf_occupancy_slots': rf_occupied,
            'yard_occupancy': len(self.containers)
        })

        return True

    def remove_container(self, unique_key):
        if unique_key not in self.container_positions: return True

        position = self.container_positions[unique_key]
        block, x, y, z, yard_type = position
        yard = self.rf_yard if yard_type == 'RF' else self.gp_yard

        if not hasattr(self, 'last_rehandled_containers'):
            self.last_rehandled_containers = []
        self.last_rehandled_containers = []

        containers_above = []
        for above_z in range(z + 1, self.yard_depth):
            above_key = yard[block, x, y, above_z]
            if above_key:
                real_key = str(above_key).replace('_2nd', '')
                if real_key not in containers_above:
                    containers_above.append(real_key)

        for above_key in reversed(containers_above):
            self.rehandling_count += 1
            self.last_rehandled_containers.append(above_key)

            if above_key in self.containers:
                info = self.containers[above_key].copy()
                size = self.get_container_size(info.get('CARGO_SIZE'))
                ctype = self.get_container_type(info.get('CARGO_TYPE'))
                
                old_pos = self.container_positions[above_key]
                self._remove_container_from_position(above_key, old_pos)

                new_pos = self.find_position_for_rehandling(size, ctype, info, old_pos)

                if new_pos:
                    self.place_container(above_key, info, new_pos)
                else:
                    self.temp_yard_containers.add(above_key)
                    self.overflow_count += 1
                    return False

        self._remove_container_from_position(unique_key, position)
        return True

    def place_container(self, unique_key, container_info, position):
        block, x, y, z, yard_type = position
        yard = self.rf_yard if yard_type == 'RF' else self.gp_yard
        size = self.get_container_size(container_info.get('CARGO_SIZE'))

        self.containers[unique_key] = container_info
        self.container_positions[unique_key] = position
        yard[block, x, y, z] = unique_key

        if size == '40ft' and x + 1 < self.yard_width:
            yard[block, x+1, y, z] = unique_key + '_2nd'

    def _remove_container_from_position(self, unique_key, position):
        if unique_key not in self.container_positions: return
        block, x, y, z, yard_type = position
        yard = self.rf_yard if yard_type == 'RF' else self.gp_yard
        
        info = self.containers.get(unique_key, {})
        size = self.get_container_size(info.get('CARGO_SIZE'))

        yard[block, x, y, z] = None
        if size == '40ft' and x + 1 < self.yard_width:
            if yard[block, x+1, y, z] == unique_key + '_2nd':
                yard[block, x+1, y, z] = None

        if unique_key in self.containers: del self.containers[unique_key]
        if unique_key in self.container_positions: del self.container_positions[unique_key]

    def update_container_times(self, time_diff_hours):
        for key in list(self.containers.keys()):
            if 'CDT_true' in self.containers[key]:
                self.containers[key]['CDT_true'] = max(0, self.containers[key]['CDT_true'] - time_diff_hours)
            if 'std_CDT_pred' in self.containers[key]:
                self.containers[key]['std_CDT_pred'] = max(0, self.containers[key]['std_CDT_pred'] - time_diff_hours)
            if 'no_std_CDT_pred' in self.containers[key]:
                self.containers[key]['no_std_CDT_pred'] = max(0, self.containers[key]['no_std_CDT_pred'] - time_diff_hours)

    def get_container_size(self, cargo_size_value):
        if pd.isna(cargo_size_value): return '20ft'
        s = str(cargo_size_value)
        if s.startswith(('45', '42', '40', 'L5')): return '40ft'
        return '20ft'

    def get_container_type(self, cargo_type_value):
        if pd.isna(cargo_type_value): return 'GP'
        return 'RF' if cargo_type_value == 'RF' else 'GP'

    def get_yard_status(self):
        gp_slots = np.count_nonzero(self.gp_yard != None)
        rf_slots = np.count_nonzero(self.rf_yard != None)
        return {
            'total_containers': len(self.containers),
            'gp_slots_occupied': gp_slots,
            'rf_slots_occupied': rf_slots,
            'rehandling_count': self.rehandling_count,
            'overflow_count': self.overflow_count
        }
