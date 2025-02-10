"""
通用工具函数

作者: zongxiang hu
创建日期: 2024-01-07
最后修改: 2024-01-07

该模块提供通用的工具函数
"""

import os
from pathlib import Path
import random
import numpy as np
import torch
from dotenv import load_dotenv
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def setup_env() -> None:
    """
    设置环境变量
    
    优先从 .env 文件加载环境变量,如果没有则使用默认值
    """
    # 加载 .env 文件
    env_path = Path(os.getenv("PROJECT_ROOT", ".")) / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"从 {env_path} 加载环境变量")
    
    # 设置 HuggingFace 镜像
    hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = hf_endpoint
    logger.info(f"设置 HuggingFace 镜像: {hf_endpoint}")


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 一些额外的设置以确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"设置随机种子: {seed}")
