"""
GroundingDINO模型封装

作者: zongxiang hu
创建日期: 2024-01-07
最后修改: 2024-01-07
"""

import torch
from typing import Dict, List, Optional, Union, Any
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.models.components.base import BaseModel
from src.utils.device import device_context
from src.utils.logger import setup_logger


class GroundingDINOModel(BaseModel):
    """GroundingDINO模型封装"""
    
    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        text: str = "person",  # 默认检测人
        box_threshold: float = 0.25,
        text_threshold: float = 0.25
    ):
        """
        初始化GroundingDINO模型
        
        Args:
            model_name: HuggingFace模型名称或本地路径
            device: 运行设备
            cache_dir: 权重缓存目录，默认使用HF_HOME
            text: 文本提示（需要小写）
            box_threshold: 检测框阈值
            text_threshold: 文本匹配阈值
        """
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(f"初始化GroundingDINO模型: {model_name}")
        
        self.text = text
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto'
        ).eval().to(device)
        
    def preprocess(
        self,
        batch: Dict[str, Union[torch.Tensor, List]]
    ) -> Dict[str, torch.Tensor]:
        """
        预处理输入数据
        
        Args:
            batch: DataLoader 返回的批次数据，包含：
                - image: torch.Tensor 或 List[PIL.Image]，批次图像数据
                - image_id: List[str]，图像ID列表
            
        Returns:
            处理后的输入数据
        """
        batch_size = len(batch['image_id'])
        self.logger.info(f"开始GroundingDINO预处理，批次大小: {batch_size}")
        
        # 准备输入
        inputs = self.processor(
            images=batch['image'],
            text=[self.text] * batch_size,  # 为每个图像复制提示词
            return_tensors="pt"
        ).to(self.device)
        
        # 打印维度信息用于调试
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                self.logger.info(f"{key} shape: {value.shape}")
        
        return inputs
        
    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, List]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        批量目标检测
        
        Args:
            batch: DataLoader 返回的批次数据，包含：
                - image: torch.Tensor 或 List[PIL.Image]，批次图像数据
                - image_id: List[str]，图像ID列表
            
        Returns:
            预测结果列表，每个元素为一张图片的检测结果，包含:
            - boxes: torch.Tensor, shape (N, 4), 边界框坐标
            - scores: torch.Tensor, shape (N,), 置信度分数
            - labels: torch.Tensor, shape (N,), 类别标签
        """
        # 预处理
        inputs = self.preprocess(batch)
        
        # 推理
        with device_context(self.device):
            outputs = self.model(**inputs)
            
        # 后处理
        return self.postprocess(outputs, batch)
        
    def postprocess(
        self,
        outputs: Any,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        后处理GroundingDINO输出
        
        Args:
            outputs: 模型原始输出
            metadata: 批次元数据，包含原始图像信息
            
        Returns:
            处理后的预测结果列表
        """
        # 获取目标尺寸
        target_sizes = [img.size[::-1] if isinstance(img, Image.Image) else img.shape[-2:] 
                       for img in metadata['image']]
                       
        # 使用processor进行后处理
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.box_threshold
        )
        
        # 转换为标准格式
        predictions = []
        for result in results:
            # 过滤结果
            keep = result["scores"] >= self.box_threshold
            boxes = result["boxes"][keep]
            scores = result["scores"][keep]
            labels = result["labels"][keep]
            
            # 添加到预测列表
            predictions.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "class_names": [self.text] * len(boxes)  # 使用相同的类别名称
            })
            
        return predictions
    