from typing import Dict, List, Union, Optional, Any
from ultralytics import YOLO
import torch
from src.models.components.base import BaseModel
from src.utils.ops import scale_boxes
from src.utils.device import device_context
from src.utils.logger import setup_logger

class YOLODetector(BaseModel):
    """YOLO检测器封装"""
    
    def __init__(
        self,
        model_name: str = "yolov8l.pt",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[int]] = None
    ):
        """
        初始化YOLO检测器
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备
            cache_dir: 缓存目录
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
            classes: 指定类别，None表示所有类别
        """
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)
        
        # 加载预训练模型
        self.model = YOLO(self.model_path)
        self.model.to(device)
        
        # 推理配置
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        
        self.logger = setup_logger(self.__class__.__name__)
        
    def preprocess(
        self, 
        batch: Dict[str, Union[torch.Tensor, List]]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """预处理输入"""
        return batch

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, List]]
    ) -> List[Dict]:
        """
        批量预测
        
        Args:
            batch: DataLoader 返回的批次数据，包含：
                - image: torch.Tensor (B, C, H, W)
                - image_id: List[str]，图像ID列表
            
        Returns:
            预测结果列表，每个元素为一张图片的检测结果，包含:
            - boxes: numpy.ndarray, shape (N, 4), 边界框坐标 (x1, y1, x2, y2)
            - scores: numpy.ndarray, shape (N,), 置信度分数
            - labels: numpy.ndarray, shape (N,), 类别标签
        """
        batch = self.preprocess(batch)

        batch_size = len(batch['image_id'])
        self.logger.info(f"开始YOLO检测，批次大小: {batch_size}")
        
        # 确保输入在正确的设备上
        images = batch['image'].to(self.device)
        
        # 批量推理
        with device_context(self.device):
            results = self.model.predict(
                images,
                conf=self.conf_thres,
                iou=self.iou_thres,
                classes=self.classes,
                verbose=False,
                stream=False  # 不使用流式处理，一次性处理整个批次
            )
            
        # 后处理
        return self.postprocess(results, batch)
        
    def postprocess(self, outputs: Any, metadata: Dict[str, Any]) -> List[Dict]:
        """
        后处理YOLO输出
        
        Args:
            outputs: YOLO模型输出
            metadata: 批次元数据，包含原始图像信息
            
        Returns:
            处理后的预测结果列表
        """
        predictions = []
        for result, orig_size in zip(outputs, metadata["metadata"].get("original_size", [])):
            boxes = result.boxes
            boxes_np = boxes.xyxy.cpu().numpy()
            
            # 如果有原始尺寸信息，将框映射回原始尺寸
            if orig_size is not None:
                # 从result中获取实际的输入尺寸
                current_size = result.orig_shape[:2]  # (H, W)
                boxes_np = scale_boxes(
                    boxes=boxes_np,
                    current_img=current_size,  # 使用实际的输入尺寸
                    original_img=tuple(orig_size),
                )
            ids = boxes.cls.cpu().numpy().astype(int)
            names = [result.names[id] for id in ids]
            prediction = {
                "boxes": boxes_np,
                "scores": boxes.conf.cpu().numpy(),
                "labels": ids,
                "class_names": names,
            }
            predictions.append(prediction)
            
        return predictions
    