import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict

# Isaac Gym 설정 파일(.yaml)에서 사용하는 커스텀 연산자 등록
# 이 부분은 지우면 안 됩니다. (특히 'resolve_default'는 G1 설정에서도 사용됨)
try:
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)
except Exception:
    # 이미 등록된 경우 발생할 수 있는 에러 방지
    pass

def make(
    seed: int, 
    task: str, 
    num_envs: int, 
    sim_device: str,
    rl_device: str,
    graphics_device_id: int = -1,
    headless: bool = False,
    multi_gpu: bool = False,
    virtual_screen_capture: bool = False,
    force_render: bool = True,
    cfg: DictConfig = None
): 
    from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator
    
    # 1. 설정값이 전달되지 않았을 경우 (직접 make 호출 시)
    if cfg is None:
        # 이미 Hydra가 초기화되어 있다면 기존 정보를 클리어하고 새로 설정
        if HydraConfig.initialized():
            hydra.core.global_hydra.GlobalHydra.instance().clear()

        # ./cfg 폴더에서 설정을 불러옴
        with initialize(config_path="./cfg"):
            # task 파라미터로 받은 태스크(예: HumanoidAMP)의 설정을 로드
            cfg = compose(config_name="config", overrides=[f"task={task}"])
            cfg_dict = omegaconf_to_dict(cfg.task)
            cfg_dict['env']['numEnvs'] = num_envs
            
    # 2. 이미 train.py 등을 통해 cfg가 넘어온 경우
    else:
        cfg_dict = omegaconf_to_dict(cfg.task)

    # 3. RL-GPU 환경 생성기 호출
    # 여기서 task_name은 HumanoidAMP가 되어야 함
    create_rlgpu_env = get_rlgames_env_creator(
        seed=seed,
        task_config=cfg_dict,
        task_name=cfg_dict["name"],
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        multi_gpu=multi_gpu,
        virtual_screen_capture=virtual_screen_capture,
        force_render=force_render,
    )
    
    return create_rlgpu_env()
