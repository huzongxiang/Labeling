"""
推理流水线组件

作者: zongxiang hu
创建日期: 2024-01-03
最后修改: 2024-01-07

该模块实现了具体的推理流水线组件,包括:
- ImagePrePipeline: 图像预处理流水线
- YOLOPostPipeline: YOLO模型后处理流水线 
- GroundingDINOPostPipeline: GroundingDINO模型后处理流水线
- SAMPostPipeline: SAM模型后处理流水线
- FlorencePostPipeline: Florence模型后处理流水线
"""

from typing import Dict, Any, Optional, Type, List
from torch.utils.data import Dataset, DataLoader
from src.pipeline.components.base import BasePrePipeline, BasePostPipeline
from src.pipeline.components.datasets import ImageDataset
from src.utils.convert import to_list

class ImagePrePipeline(BasePrePipeline):
    """图像预处理流水线"""
    
    def __init__(
        self,
        dataset_cls: Optional[Type[Dataset]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs
    ):
        """
        初始化图像预处理流水线
        
        Args:
            dataset_cls: 数据集类,默认为ImageDataset
            batch_size: 批次大小,默认为1
            num_workers: 数据加载线程数,默认为0
            **kwargs: 传递给数据集的额外参数
        """
        super().__init__()
        self.dataset_cls = dataset_cls or ImageDataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
    
    def setup(self, data: Dict[str, Any]) -> DataLoader:
        """
        设置数据集并返回数据加载器
        
        Args:
            data: 包含data_file路径的字典
            
        Returns:
            配置好的DataLoader实例
        """
        data_file = data.get("data_file")
        dataset = self.dataset_cls(
            data_file=data_file,
            **self.kwargs
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.logger.info(f"数据集设置完成，样本数量: {len(dataset)}")
        return dataloader

class PostPipeline(BasePostPipeline):
    """YOLO模型后处理流水线"""
    
    def format_predictions(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化YOLO预测结果
        
        Args:
            pred: 原始预测结果字典，包含boxes、scores、labels等
            
        Returns:
            格式化后的预测结果字典
        """
        return {
            "boxes": to_list(pred["boxes"]),
            "scores": to_list(pred["scores"]),
            "class_id": to_list(pred["labels"])
        }
    

class YOLOPostPipeline(BasePostPipeline):
    """YOLO模型后处理流水线"""
    
    def format_predictions(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化YOLO预测结果
        
        Args:
            pred: 原始预测结果字典，包含boxes、scores、labels等
            
        Returns:
            格式化后的预测结果字典
        """
        return {
            "boxes": to_list(pred["boxes"]),
            "scores": to_list(pred["scores"]),
            "class_id": to_list(pred["labels"]),
            "class_names": pred["class_names"],
        }


class GroundingDINOPostPipeline(BasePostPipeline):
    """GroundingDINO模型后处理流水线"""
    
    def format_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        格式化GroundingDINO预测结果
        
        Args:
            predictions: 原始预测结果列表
            
        Returns:
            格式化后的预测结果列表
        """
        return [
            {
                "bbox": pred["bbox"],
                "score": pred["score"],
                "class_name": pred["class_name"]
            }
            for pred in predictions
        ]

class SAMPostPipeline(BasePostPipeline):
    """SAM模型后处理流水线"""
    
    def format_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        格式化SAM预测结果
        
        Args:
            predictions: 原始预测结果列表
            
        Returns:
            格式化后的预测结果列表
        """
        return [
            {
                "segmentation": pred["segmentation"],
                "bbox": pred["bbox"],
                "score": pred["score"],
                "area": pred["area"]
            }
            for pred in predictions
        ]

    
DefaultPrePipeline = ImagePrePipeline
DefaultPostPipeline = PostPipeline
