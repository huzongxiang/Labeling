"""
图像转换工具

作者: zongxiang hu
创建日期: 2024-01-07
最后修改: 2024-01-07

该模块提供图像转换相关的工具类,包括:
- YOLOTransform: YOLO模型所需的图像和标签转换
- ImageTransform: 通用图像转换
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from typing import Union, Tuple, List, Optional
from src.utils.ops import LetterBox

class YOLOTransform:
    """YOLO模型所需的图像和标签转换"""
    
    def __init__(
        self,
        target_size: Union[int, Tuple[int, int]] = 640,
        scaleup: bool = False
    ):
        """
        初始化转换器
        
        Args:
            target_size: 目标尺寸，可以是单个整数或(height, width)元组
            scaleup: 是否放大小图片
        """
        self.letterbox = LetterBox(new_shape=target_size, scaleup=scaleup)
        self.transform = transforms.ToTensor()
        
    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        boxes: Optional[np.ndarray] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        """
        转换图像和标签
        
        Args:
            image: 输入图像，可以是:
                   - torch.Tensor (H, W, C) 或 (C, H, W)
                   - numpy.ndarray (H, W, C)
            boxes: 边界框坐标 (N, 4) [x1, y1, x2, y2]，可选
                   
        Returns:
            如果没有boxes: 返回转换后的图像 torch.Tensor (C, H, W)，值范围 [0, 1]
            如果有boxes: 返回 (转换后的图像, 转换后的边界框坐标)
        """
        if boxes is None:
            # 只处理图像
            image = self.letterbox(image=image)
            image = self.transform(image)
            return image, None
        else:
            # 同时处理图像和标签
            image, boxes = self.letterbox(image=image, boxes=boxes)
            image = self.transform(image)
            return image, boxes
        
class ImageTransform:
    """通用图像转换"""
    
    def __init__(
        self,
        target_size: Union[int, Tuple[int, int]],
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        """
        初始化转换器
        
        Args:
            target_size: 目标尺寸，可以是单个整数或(height, width)元组
            mean: 归一化均值
            std: 归一化标准差
        """
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
            transforms.Resize(
                target_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        转换图像
        
        Args:
            image: 输入图像，可以是:
                   - torch.Tensor (H, W, C)
                   - numpy.ndarray (H, W, C)
                   
        Returns:
            转换后的图像 torch.Tensor (C, H, W)，已归一化
        """
        return self.transform(image)
