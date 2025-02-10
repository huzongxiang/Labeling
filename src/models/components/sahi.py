from typing import Dict, List, Union, Optional, Any
import torch
from torchvision import transforms
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import Path
from src.models.components.base import BaseModel
from src.utils.device import device_context
from src.utils.logger import setup_logger

class SahiYOLODetector(BaseModel):
    """SAHI + YOLO 检测器封装"""
    
    def __init__(
        self,
        model_name: str = "yolov8l.pt",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        conf_thres: float = 0.25,
        slice_size: int = 512,
        overlap_ratio: float = 0.2,
        classes: Optional[List[int]] = None,
    ):
        """
        初始化SAHI YOLO检测器
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备
            cache_dir: 缓存目录
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
            slice_size: 切片尺寸
            overlap_ratio: 切片重叠比
            classes: 指定类别，None表示所有类别
        """
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)
        
        # 使用 SAHI 封装 YOLO 模型
        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=Path(self.model_path),
            confidence_threshold=conf_thres,
            device=device
        )
        
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.classes = classes
        
        self.logger = setup_logger(self.__class__.__name__)
        
    def preprocess(
        self, 
        batch: Dict[str, Union[torch.Tensor, List]]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        预处理输入
        
        Args:
            batch: DataLoader 返回的批次数据，包含：
                - image: torch.Tensor (B, C, H, W)
                - image_id: List[str]，图像ID列表
                
        Returns:
            处理后的批次数据
        """
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
                - image_id: str，图像ID
                - boxes: List[List[float]]，边界框坐标 [x1, y1, x2, y2]
                - scores: List[float]，置信度分数
                - labels: List[int]，类别ID
                - class_names: List[str]，类别名称
        """
        batch = self.preprocess(batch)
        batch_size = len(batch['image_id'])
        self.logger.info(f"开始SAHI YOLO检测，批次大小: {batch_size}")
        
        results = []
        
        for idx, img in enumerate(batch['image']):
            img_id = batch['image_id'][idx]
            # 将图像转换为PIL格式（SAHI需要PIL输入）
            img_pil = transforms.ToPILImage()(img.cpu())
            
            # SAHI 切片推理
            with device_context(self.device):
                prediction = get_sliced_prediction(
                    img_pil,
                    self.model,
                    slice_height=self.slice_size,
                    slice_width=self.slice_size,
                    overlap_height_ratio=self.overlap_ratio,
                    overlap_width_ratio=self.overlap_ratio,
                    verbose=0,
                )
                
            # 转换预测结果
            results.append(self.postprocess(prediction, img_id))
        
        return results

    def postprocess(self, prediction: Any, img_id: str) -> Dict:
        """
        后处理SAHI输出
        
        Args:
            prediction: SAHI的预测结果
            img_id: 图像ID
        
        Returns:
            单张图像的检测结果，包含:
                - image_id: str，图像ID
                - boxes: List[List[float]]，边界框坐标 [x1, y1, x2, y2]
                - scores: List[float]，置信度分数
                - labels: List[int]，类别ID
                - class_names: List[str]，类别名称
        """
        boxes = []
        scores = []
        labels = []
        class_names = []
        
        for obj in prediction.object_prediction_list:
            # 如果指定了类别且当前类别不在指定范围内，则跳过
            if self.classes is not None and obj.category.id not in self.classes:
                continue
                
            boxes.append(obj.bbox.to_xyxy())
            scores.append(obj.score.value)
            labels.append(obj.category.id)
            class_names.append(obj.category.name)
        
        return {
            "image_id": img_id,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "class_names": class_names,
        }
    