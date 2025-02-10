"""
标注模块

作者: zongxiang hu
创建日期: 2024-01-03
最后修改: 2024-01-03

该模块实现了标注模块的主要逻辑,包括:
- 模型推理和后处理
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from src.models.components import BaseModel
from src.utils.logger import setup_logger

class LabelingModule(nn.Module):
    """模型模块"""
    
    def __init__(
        self,
        model: BaseModel,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化模型模块
        
        Args:
            model: 模型实例
            config: 配置信息
        """
        super().__init__()
        self.model = model
        self.config = config or {}
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(f"初始化检测模块，使用模型: {model.__class__.__name__}")
        self.logger.debug(f"配置信息: {config}")
        
    def forward(
        self, 
        batch: Tuple[torch.Tensor, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            batch: dataloader加载的批次数据，包含:
                - images: 图像张量 (B, C, H, W)
                - metadata: 批次元数据
            
        Returns:
            预测结果列表，每个结果包含:
                - boxes: 检测框坐标 [x1, y1, x2, y2]
                - scores: 置信度分数
                - labels: 类别标签
        """
        results = self.model(batch)
        return {"predictions": results, "metadata": batch}
    