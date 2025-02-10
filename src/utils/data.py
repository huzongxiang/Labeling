"""
数据处理工具

作者: zongxiang hu
创建日期: 2024-01-06
最后修改: 2024-01-06

该模块提供数据处理相关的工具函数和类,包括:
- DataInfo: 数据源的基本信息类
- Sample: 数据集样本类
- 数据验证函数
- 数据转换函数
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union, Any
from pathlib import Path
import json
import yaml
import torch
import numpy as np

@dataclass
class DataInfo:
    """数据源的基本信息"""
    name: str  # 数据源名称
    type: str  # 数据类型：image, video, dataset等
    format: str  # 数据格式：jpg, mp4, zip等
    size: Optional[int] = None  # 数据大小（字节）
    metadata: Optional[Dict] = None  # 其他元数据

@dataclass
class Sample:
    """数据集样本类"""
    # 图像数据
    image: Union[torch.Tensor, np.ndarray]  # 图像数据 (H, W, C) for numpy, (C, H, W) for tensor
    image_id: Union[str, int]  # 图像ID
    image_path: Union[str, Path]  # 图像路径
    
    # 可选的标注数据
    label: Optional[Dict[str, Any]] = None  # 标注信息
    metadata: Optional[Dict[str, Any]] = None  # 其他元数据
    
    def validate(self) -> bool:
        """
        验证样本格式
        
        Returns:
            是否为有效的样本格式
        
        Raises:
            ValueError: 如果样本格式不正确，包含具体的错误信息
        """
        # 验证图像
        if isinstance(self.image, np.ndarray):
            if self.image.ndim != 3:  # (H, W, C)
                raise ValueError(f"图像维度错误: 期望3维(H,W,C)，实际为{self.image.ndim}维")
            if self.image.shape[2] != 3:  # RGB
                raise ValueError(f"图像通道数错误: 期望3通道，实际为{self.image.shape[2]}通道")
        elif isinstance(self.image, torch.Tensor):
            if self.image.dim() != 3:  # (C, H, W)
                raise ValueError(f"图像维度错误: 期望3维(C,H,W)，实际为{self.image.dim()}维")
            if self.image.size(0) not in [1, 3]:  # 通道数
                raise ValueError(f"图像通道数错误: 期望1或3通道，实际为{self.image.size(0)}通道")
        else:
            raise ValueError(f"图像类型错误: 期望 np.ndarray 或 torch.Tensor，实际为 {type(self.image)}")
            
        # 验证图像ID
        if not isinstance(self.image_id, (str, int)):
            raise ValueError(f"图像ID类型错误: 期望 str 或 int，实际为 {type(self.image_id)}")
            
        # 验证图像路径
        if not isinstance(self.image_path, (str, Path)):
            raise ValueError(f"图像路径类型错误: 期望 str 或 Path，实际为 {type(self.image_path)}")
            
        # 验证标注(如果存在)
        if self.label is not None and not isinstance(self.label, dict):
            raise ValueError(f"标注类型错误: 期望 dict，实际为 {type(self.label)}")
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，返回所有非空属性
        
        Returns:
            包含所有非空样本属性的字典，包括：
            - image: 图像数据
            - image_id: 图像ID
            - image_path: 图像路径
            - label: 标注信息（如果存在）
            - metadata: 其他元数据（如果存在）
        """
        result = {
            "image": self.image,
            "image_id": self.image_id,
            "image_path": str(self.image_path)
        }
        
        # 只添加非空属性
        if self.label is not None:
            result["label"] = self.label
            
        if self.metadata is not None:
            result["metadata"] = self.metadata
            
        return result
        
    @classmethod
    def from_dict(cls, image: Union[torch.Tensor, np.ndarray], metadata: Dict[str, Any]) -> 'Sample':
        """
        从字典创建样本
        
        Args:
            image: 图像数据
            metadata: 元数据字典
            
        Returns:
            Sample实例
        """
        return cls(
            image=image,
            image_id=metadata["image_id"],
            image_path=metadata["image_path"],
            label=metadata.get("label")
        )

def load_data_info(path: Union[str, Path]) -> Dict:
    """
    加载数据信息文件(支持json和yaml格式)
    
    Args:
        path: 数据信息文件路径
        
    Returns:
        数据信息字典
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
        
    if path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

def save_data_info(data: Dict, path: Union[str, Path]) -> None:
    """
    保存数据信息到文件
    
    Args:
        data: 数据信息字典
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif path.suffix in ['.yaml', '.yml']:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

def get_file_info(path: Union[str, Path]) -> DataInfo:
    """
    获取文件的基本信息
    
    Args:
        path: 文件路径
        
    Returns:
        DataInfo实例
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
        
    return DataInfo(
        name=path.stem,
        type=path.suffix[1:],  # 去掉点号
        format=path.suffix[1:],
        size=path.stat().st_size,
        metadata={
            "created": path.stat().st_ctime,
            "modified": path.stat().st_mtime
        }
    ) 