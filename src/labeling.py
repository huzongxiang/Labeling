"""
标注流水线主入口

作者: zongxiang hu
创建日期: 2024-01-03
最后修改: 2024-01-06

该模块实现了自动标注流水线的主要逻辑,包括:
- 数据加载和预处理
- 模型推理和后处理
- 结果保存
"""

import hydra
import pyrootutils
from omegaconf import DictConfig, OmegaConf

# 设置项目根目录
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.logger import setup_logger
from src.utils.utils import set_seed, setup_env

logger = setup_logger("main")


def labeling(cfg: DictConfig) -> None:
    """
    执行标注流程
    
    Args:
        cfg: Hydra配置，包含:
            - data: 数据模块配置
            - model: 模型配置
            - pipeline: 流水线配置
    """
    # 设置随机种子
    if cfg.get("seed"):
        set_seed(cfg.seed)
    
    try:
        # 实例化数据模块
        logger.info(f"实例化数据模块 <{cfg.data._target_}>")
        data_module = hydra.utils.instantiate(cfg.data)
        
        # 实例化标注模块
        logger.info(f"实例化标注模块 <{cfg.model._target_}>")
        labeling_module = hydra.utils.instantiate(cfg.model)
        
        # 实例化流水线
        logger.info(f"实例化流水线 <{cfg.pipeline._target_}>")
        pipeline = hydra.utils.instantiate(
            cfg.pipeline,
            data_module=data_module,
            labeling_module=labeling_module
        )
        
        # 执行流水线
        logger.info("开始执行流水线...")
        pipeline.run()
        
    except Exception as e:
        logger.error(f"处理出错: {str(e)}", exc_info=True)
        raise e


@hydra.main(version_base="1.3", config_path="../configs", config_name="labeling")
def main(cfg: DictConfig) -> None:
    """
    标注流程入口
    
    Args:
        cfg: Hydra配置
    """
    # 设置环境变量
    setup_env()
    
    # 打印配置信息
    logger.info(f"配置信息:\n{OmegaConf.to_yaml(cfg)}")
    
    # 执行标注流程
    labeling(cfg)


if __name__ == "__main__":
    main()
