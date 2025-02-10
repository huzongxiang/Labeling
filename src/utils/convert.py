"""
数据转换工具

作者: zongxiang hu
创建日期: 2024-01-07
最后修改: 2024-01-07

该模块提供通用的数据转换工具函数
"""

from typing import Any, Dict, List
from pathlib import Path
import torch
import numpy as np

def to_list(data: Any) -> Any:
    """
    将数据转换为Python原生类型
    
    Args:
        data: 输入数据，可以是tensor、ndarray、Path等
        
    Returns:
        转换后的Python原生类型数据
    """
    if torch.is_tensor(data):
        return data.cpu().numpy().tolist()
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, Path):
        return str(data)
    return data
    
def convert_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    转换批次数据为Python原生类型
    
    Args:
        batch: 批次数据字典
        
    Returns:
        转换后的批次数据字典
    """
    return {
        key: to_list(value)
        for key, value in batch.items()
    } 