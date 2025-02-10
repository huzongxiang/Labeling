from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
from pathlib import Path
from torch.utils.data import DataLoader
from src.utils.logger import setup_logger
from src.utils.file import ResultWriter
from src.utils.convert import convert_batch

class BasePrePipeline(ABC):
    """预处理流水线基类"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    @abstractmethod
    def setup(self, data: Dict[str, Any]) -> DataLoader:
        """设置数据集并返回数据加载器"""
        pass
        
    def __call__(self, data: Dict[str, Any]) -> DataLoader:
        """调用预处理流水线"""
        return self.setup(data)

class BasePostPipeline(ABC):
    """后处理流水线基类"""
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        save_name: str = "results.json"
    ):
        """
        初始化后处理流水线
        
        Args:
            save_dir: 保存目录
            save_name: 保存文件名
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.save_dir = Path(save_dir)
        self.save_path = self.save_dir / save_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = ResultWriter(self.save_path)
        
    @abstractmethod
    def format_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """格式化预测结果，由子类实现"""
        pass
        
    def process(self, outputs: Dict[str, Any]) -> None:
        """处理模型输出"""
        predictions = outputs.get("predictions", [])
        metadata = outputs.get("metadata", {})
        
        if not predictions or not metadata:
            self.logger.warning("没有预测结果或元数据")
            return
            
        # 转换元数据
        batch_metadata = convert_batch(metadata)
        
        # 组织批次结果
        batch_size = len(predictions)
        batch_results = []
        
        for i in range(batch_size):
            result = {
                "image_path": str(batch_metadata["image_path"][i]),
                "image_id": str(batch_metadata["image_id"][i]),
                "predictions": self.format_predictions(predictions[i])
            }
            batch_results.append(result)
        
        # 写入批次结果
        with self.writer as writer:
            writer.write(batch_results)
            
    def __call__(self, outputs: Dict[str, Any]) -> None:
        """调用后处理流水线"""
        self.process(outputs)
        