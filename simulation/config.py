from dataclasses import dataclass

@dataclass
class YardSimConfig:
    """
    야드 시뮬레이션 관련 설정들을 통합 관리하는 구성(Configuration) 클래스.
    실험(run_experiment)과 시각화(run_visualization) 모두에서 공유되어 사용됩니다.
    """
    width: int = 40
    height: int = 12
    depth: int = 5
    gp_blocks: int = 2
    rf_blocks: int = 2
    
    # CDT 기준점 ('std_CDT_pred', 'no_std_CDT_pred', 'CDT_true' 등)
    cdt_key: str = 'std_CDT_pred'
    
    # 시뮬레이션 중 상태 업데이트 무시 여부 (Exp 4 용도 등)
    ignore_updates: bool = False
