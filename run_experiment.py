import pandas as pd
import numpy as np
import os
import argparse
from multiprocessing import Pool
import time

from simulation.config import YardSimConfig
from simulation.core import YardSimulation, RandomStrategy, FloorByFloorStrategy, GlobalSearchStrategy

# ======================================================================
# [글로벌 변수] 워커 프로세스 내 데이터 캐싱 (메모리 최적화)
# ======================================================================
worker_data = None 

def worker_init(file_path):
    global worker_data
    try:
        worker_data = pd.read_csv(file_path, parse_dates=['EVENT_TS'])
    except Exception as e:
        print(f"[Worker Error] Failed to load data: {e}")

# ======================================================================
# [유틸리티] 노이즈 함수
# ======================================================================
def add_noise_to_data(df: pd.DataFrame, noise_level: float) -> pd.DataFrame:
    df_noisy = df.copy()
    if noise_level == 0.0:
        df_noisy['std_CDT_pred'] = df_noisy['CDT_true']
    else:
        options = np.array([noise_level, -noise_level])
        selected_noise = np.random.choice(options, size=len(df_noisy))
        df_noisy['std_CDT_pred'] = df_noisy['CDT_true'] * (1.0 + selected_noise)
    df_noisy['std_CDT_pred'] = df_noisy['std_CDT_pred'].clip(lower=1.0)
    return df_noisy

# ======================================================================
# [Task 실행] 통합 워커 함수
# ======================================================================
def _execute_single_task(task_info):
    global worker_data
    if worker_data is None:
        return {'Error': 'Worker data not loaded'}

    config_params = task_info['config']
    exp_id = task_info['exp_id']
    strategy_name = task_info['strategy_name']
    cdt_source = task_info['cdt_source']
    noise_level = task_info.get('noise_level', None)
    ignore_updates = task_info.get('ignore_updates', False)
    repetition = task_info.get('repetition', 1)
    
    config = YardSimConfig(
        width=config_params['width'],
        height=config_params['height'],
        depth=config_params['depth'],
        gp_blocks=config_params['gp_blocks'],
        rf_blocks=config_params['rf_blocks'],
        cdt_key=cdt_source,
        ignore_updates=ignore_updates
    )
    
    run_data = worker_data.copy()
    if noise_level is not None and noise_level > 0.0:
        run_data = add_noise_to_data(run_data, noise_level)
    
    # 전략 주입
    strategy = None
    if strategy_name == 'Random':
        strategy = RandomStrategy() # Random은 cdt_key 없이 동작
    elif strategy_name == 'Floor':
        strategy = FloorByFloorStrategy(cdt_key=cdt_source)
    elif strategy_name == 'Global':
        strategy = GlobalSearchStrategy(cdt_key=cdt_source)
        
    sim = YardSimulation(
        yard_config=config,
        stacking_strategy=strategy
    )
    
    _, rehandling, stats = sim.run(run_data)
    
    exp_label = f"Exp [{exp_id}]"
    if exp_id == 2:
        exp_label += f" Smart (Binary +/- {noise_level})" if noise_level else " Random (Baseline)"

    result_dict = {
        'Repetition': repetition,
        'GP_Blocks': config.gp_blocks,
        'Experiment': exp_label, 
        'Strategy': strategy.__class__.__name__,
        'CDT_Key': cdt_source,
        'Rehandling': rehandling,
        'Noise_Sigma': noise_level if noise_level is not None else np.nan,
        **stats
    }
    return result_dict

# ======================================================================
# 메인 실행 
# ======================================================================
def main(args):
    DATA_FILENAME = "datas/sim_data_251208_v2.csv" 
    if not os.path.exists(DATA_FILENAME):
        print(f"ERROR: '{DATA_FILENAME}' not found.")
        return
    
    gp_block_list = args.gp_blocks
    experiments_to_run = args.experiments
    if 'all' in experiments_to_run:
        experiments_to_run = ['1']  # 테스트 시 기본값 가볍게
    
    num_cores = args.workers
    repeat_count = args.repeat
    
    if args.output and os.path.exists(args.output):
        try: os.remove(args.output)
        except: pass
            
    all_tasks = []
    
    for r in range(repeat_count):
        current_rep = r + 1
        for block_count in gp_block_list:
            cfg = {
                'width': 24, 'height': 10, 'depth': 5, 
                'gp_blocks': block_count, 'rf_blocks': 2
            }
            if '1' in experiments_to_run:
                all_tasks.append({'repetition': current_rep, 'config': cfg, 'exp_id': 1, 'strategy_name': 'Random', 'cdt_source': 'None'})
                all_tasks.append({'repetition': current_rep, 'config': cfg, 'exp_id': 1, 'strategy_name': 'Floor', 'cdt_source': 'std_CDT_pred'})
                all_tasks.append({'repetition': current_rep, 'config': cfg, 'exp_id': 1, 'strategy_name': 'Floor', 'cdt_source': 'CDT_true'})

    print(f" -> Total {len(all_tasks)} tasks generated")
    
    start_time = time.time()
    results = []
    with Pool(processes=num_cores, initializer=worker_init, initargs=(DATA_FILENAME,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_execute_single_task, all_tasks), 1):
            results.append(res)
            print(f"   [Progress] {i}/{len(all_tasks)} finished.")

    print(f" -> Finished in {time.time() - start_time:.2f} seconds.")

    if args.output:
        df_res = pd.DataFrame(results)
        df_res.to_csv(args.output, index=False)
        print(f" -> Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiments', nargs='*', default=['all'])
    parser.add_argument('-g', '--gp_blocks', nargs='*', default=[4], type=int)
    parser.add_argument('-o', '--output', type=str, default='experiment_results.csv')
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-r', '--repeat', type=int, default=1)
    args = parser.parse_args()
    main(args)
