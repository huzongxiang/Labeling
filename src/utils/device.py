"""
设备上下文管理器

作者: zongxiang hu
创建日期: 2024-01-07
最后修改: 2024-01-07
"""

from typing import Optional
from contextlib import contextmanager
import torch

@contextmanager
def device_context(device: str = "cuda", dtype: Optional[torch.dtype] = None):
    """
    设备上下文管理器，根据设备类型自动选择合适的上下文
    
    Args:
        device: 设备类型，'cuda' 或 'cpu'
        dtype: 数据类型，仅在使用CUDA时有效
        
    Yields:
        上下文管理器
    """
    if device == "cuda" and dtype is not None:
        ctx = torch.autocast(device, dtype=dtype)
    else:
        ctx = torch.inference_mode()
        
    with ctx:
        yield 