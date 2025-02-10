"""
图像操作工具

作者: zongxiang hu
创建日期: 2024-01-07
最后修改: 2024-01-07

该模块提供基本的图像操作工具,包括:
- LetterBox: 图像和标签的等比例缩放并填充
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional
import torch

def scale_boxes(
    boxes: np.ndarray,
    current_img: Union[np.ndarray, Tuple[int, int]],
    original_img: Union[np.ndarray, Tuple[int, int]]
) -> np.ndarray:
    """
    将预测框从当前尺寸映射回原始图像尺寸
    
    Args:
        boxes: 预测框坐标 (N, 4) [x1, y1, x2, y2]
        current_img: 当前图像或其尺寸 (H, W, C) 或 (H, W)
        original_img: 原始图像或其尺寸 (H, W, C) 或 (H, W)
        
    Returns:
        映射后的预测框坐标 (N, 4)
    """
    if len(boxes) == 0:
        return boxes
        
    # 获取尺寸
    if isinstance(current_img, (tuple, list, np.ndarray)):
        current_shape = current_img[:2] if isinstance(current_img, np.ndarray) else current_img
    else:
        raise ValueError("current_img必须是numpy数组、元组或列表")
        
    if isinstance(original_img, (tuple, list, np.ndarray)):
        original_shape = original_img[:2] if isinstance(original_img, np.ndarray) else original_img
    else:
        raise ValueError("original_img必须是numpy数组、元组或列表")
        
    # 转换为numpy数组以确保计算正确
    current_shape = np.array(current_shape)
    original_shape = np.array(original_shape)
    
    # 计算缩放比例
    ratio = min(current_shape[0] / original_shape[0], 
                current_shape[1] / original_shape[1])
            
    # 计算填充
    new_unpad = (int(round(original_shape[1] * ratio)), 
                 int(round(original_shape[0] * ratio)))
    pad_w = (current_shape[1] - new_unpad[0]) / 2
    pad_h = (current_shape[0] - new_unpad[1]) / 2
    
    # 复制并处理boxes
    boxes = boxes.copy()
    
    # 减去填充
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    
    # 缩放回原始尺寸
    boxes[:, [0, 2]] /= ratio
    boxes[:, [1, 3]] /= ratio
    
    # 裁剪到图像边界
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_shape[0])
    
    return boxes


class LetterBox:
    """图像和标签的等比例缩放并填充"""
    
    def __init__(
        self,
        new_shape: Union[int, Tuple[int, int]] = (640, 640),
        scaleup: bool = True
    ):
        """
        初始化
        
        Args:
            new_shape: 目标尺寸，可以是单个整数或(height, width)元组
            scaleup: 是否允许放大
        """
        self.new_shape = new_shape if isinstance(new_shape, tuple) else (new_shape, new_shape)
        self.scaleup = scaleup
        
    def _get_transform_params(self, shape: Tuple[int, int]) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
        """
        计算转换参数
        
        Args:
            shape: 原始尺寸 (height, width)
            
        Returns:
            缩放比例 r
            新的未填充尺寸 new_unpad (width, height)
            填充信息 (left, top)
        """
        # 计算缩放比例
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # 只缩小，不放大
            r = min(r, 1.0)
            
        # 计算新的尺寸
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        # 计算填充
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        return r, new_unpad, (int(round(dw - 0.1)), int(round(dh - 0.1)))
        
    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        boxes: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        等比例缩放并填充
        
        Args:
            image: 输入图像 (H, W, C)，可选
            boxes: 边界框坐标 (N, 4) [x1, y1, x2, y2]，可选
            
        Returns:
            如果只有image: 返回处理后的图像 (H, W, C)
            如果只有boxes: 返回处理后的边界框坐标 (N, 4)
            如果都有: 返回 (处理后的图像, 处理后的边界框坐标)
        """
        if image is None and boxes is None:
            raise ValueError("必须提供image或boxes中的至少一个参数")
            
        shape = image.shape[:2] if image is not None else boxes.shape[:2]  # [height, width]
        r, new_unpad, (left, top) = self._get_transform_params(shape)
        
        result = []
        
        # 处理图像
        if image is not None:
            # 等比例缩放
            if shape[::-1] != new_unpad:
                image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
                
            # 添加灰色填充
            image = cv2.copyMakeBorder(
                image, top, self.new_shape[0]-new_unpad[1]-top,
                left, self.new_shape[1]-new_unpad[0]-left,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            )
            result.append(image)
            
        # 处理边界框
        if boxes is not None:
            boxes_copy = boxes.copy()
            if len(boxes_copy):
                boxes_copy[:, [0, 2]] = boxes_copy[:, [0, 2]] * r + left
                boxes_copy[:, [1, 3]] = boxes_copy[:, [1, 3]] * r + top
            result.append(boxes_copy)
            
        return result[0] if len(result) == 1 else tuple(result)
    