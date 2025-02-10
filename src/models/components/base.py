from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import torch.nn as nn
from huggingface_hub import snapshot_download
from pathlib import Path

class BaseModel(nn.Module, ABC):
    """模型基类"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_dir: Optional[str] = None
    ):
        """
        初始化模型基类
        
        Args:
            model_name: HuggingFace模型名称
            device: 运行设备
            cache_dir: 权重缓存目录，默认使用HF_HOME
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_path = self._get_model_path()
        
    def _get_model_path(self) -> str:
        """
        获取模型路径，按以下顺序检查：
        1. 如果是 YOLO 模型，直接检查 cache_dir/YOLO/{model_name}.pt
        2. 否则使用 snapshot_download 下载并获取实际的模型路径
        
        Returns:
            模型路径
        """            
        # YOLO 模型特殊处理
        if 'yolo' in self.model_name.lower():
            yolo_path = Path(self.cache_dir) / "YOLO" / f"{self.model_name}"
            return str(yolo_path)
            
        # 其他模型使用 snapshot_download
        transformers_path = Path(self.cache_dir) / "TRANSFORMERS" / self.model_name
        try:
            model_path = snapshot_download(
                repo_id=self.model_name,
                cache_dir=transformers_path,
            )
            return model_path
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {self.model_name}: {e}")
    
    @abstractmethod
    def preprocess(self, inputs: Any) -> Any:
        """预处理输入"""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """模型推理"""
        pass
        
    @abstractmethod
    def postprocess(self, outputs: Any, metadata: Dict[str, Any]) -> Any:
        """
        后处理模型输出
        
        Args:
            outputs: 模型原始输出
            metadata: 批次元数据
            
        Returns:
            后处理的结果
        """
        pass

    
