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
            return_tensors="pt",
            do_rescale=False  # 如果输入图像已经归一化，避免重复rescale
        ).to(self.device)
        
        # 确保输入张量形状正确
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].reshape(
                batch_size, -1, inputs['pixel_values'].shape[-2], inputs['pixel_values'].shape[-1]
            )
        
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
            
        # 后处理时传递inputs参数
        return self.postprocess(outputs, batch, inputs)  # 新增inputs参数
        
    def postprocess(
        self,
        outputs: Any,
        metadata: Dict[str, Any],
        inputs: Dict[str, torch.Tensor]  # 新增输入参数
    ) -> List[Dict[str, torch.Tensor]]:
        """
        后处理GroundingDINO输出
        
        Args:
            outputs: 模型原始输出
            metadata: 批次元数据，包含原始图像信息
            inputs: 预处理输入数据
            
        Returns:
            处理后的预测结果列表
        """
        # 获取目标尺寸
        target_sizes = [img.size[::-1] if isinstance(img, Image.Image) else img.shape[-2:] 
                       for img in metadata['image']]
                       
        # 使用processor进行后处理时添加input_ids参数
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],  # 添加input_ids参数
            target_sizes=target_sizes,
            text_threshold=self.text_threshold,
            box_threshold=self.box_threshold
        )
        
        # 转换为标准格式
        predictions = []
        for result in results:
            # 确保使用布尔掩码进行过滤
            keep = (result["scores"] >= self.box_threshold).bool()  # 转换为布尔张量
            
            # 转换为正确类型的张量
            boxes = result["boxes"][keep].detach().cpu()
            scores = result["scores"][keep].detach().cpu()
            
            # 生成整数类型的标签（假设所有检测结果都属于同一类别）
            labels = torch.zeros(len(boxes), dtype=torch.int64)  # 创建整数标签
            
            # 添加到预测列表
            predictions.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,  # 使用生成的整数标签
                "class_names": [self.text] * len(boxes)
            })
            
        return predictions
    