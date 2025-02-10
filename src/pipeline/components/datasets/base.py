from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T
from src.utils.file import load_json
from src.utils.logger import setup_logger
from src.utils.data import Sample
import cv2
import numpy as np

class BaseDataset(Dataset, ABC):
    """数据集基类"""
    
    def __init__(
        self,
        data_file: Union[str, Path],
        transform: Optional[T.Compose] = None,
        target_transform: Optional[T.Compose] = None
    ):
        """
        初始化数据集
        
        Args:
            data_file: data.json文件路径
            transform: 图像变换
            target_transform: 标签变换
        """
        super().__init__()
        self.data_file = Path(data_file)
        self.transform = transform
        self.target_transform = target_transform
        self.logger = setup_logger(self.__class__.__name__)
        
        # 子类需要实现的属性
        self.samples: List[Dict[str, Any]] = []
        
    @abstractmethod
    def _load_data(self) -> None:
        """加载数据集信息"""
        pass
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            Sample实例,包含:
            - image: 预处理后的图像 (PIL.Image或torch.Tensor)
            - image_id: 图像ID
            - image_path: 图像路径
            - original_size: 原始图像大小 (H, W)
            - label: 可选的标注信息
        
        Raises:
            ValueError: 如果样本格式不正确
        """
        pass
        
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        使用 OpenCV 加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            numpy.ndarray: BGR格式的图像数组
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            return image
        except Exception as e:
            self.logger.error(f"加载图像失败 {image_path}: {str(e)}")
            raise e

class ImageDataset(BaseDataset):
    """基础图像数据集"""
    
    def __init__(
        self,
        data_file: Union[str, Path],
        transform: Optional[T.Compose] = None,
        target_transform: Optional[T.Compose] = None
    ):
        """
        初始化数据集
        
        Args:
            data_file: data.json文件路径
            transform: 图像变换
            target_transform: 标签变换
        """
        super().__init__(data_file, transform, target_transform)
        # 加载数据
        self._load_data()
        
    def _load_data(self) -> None:
        """从data.json加载数据集信息"""
        data_info = load_json(self.data_file)
        if "images_dir" not in data_info:
            raise ValueError("data.json必须包含'images_dir'字段")
            
        # 获取图片文件
        images_dir = Path(data_info["images_dir"])
        if not images_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {images_dir}")
            
        image_files = sorted(
            sum([list(images_dir.glob(ext)) for ext in ["*.jpg", "*.jpeg", "*.png"]], [])
        )
        if not image_files:
            raise ValueError(f"在目录 {images_dir} 中没有找到图片文件")
            
        # 加载标注
        annotations = {}
        if "annotations_file" in data_info:
            annotations_file = Path(data_info["annotations_file"])
            if annotations_file.exists():
                annotations = load_json(annotations_file)
                self.logger.info(f"加载标注文件: {annotations_file}")
            
        # 构建样本列表
        self.samples = []
        for img_path in image_files:
            sample = {"image_path": img_path, "image_id": img_path.stem}
            if img_path.stem in annotations:
                sample["label"] = annotations[img_path.stem]
            self.samples.append(sample)
        
        self.logger.info(f"加载了 {len(self.samples)} 个图像，其中 {len(annotations)} 个有标注")
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        sample_dict = self.samples[idx]
        image = self._load_image(sample_dict["image_path"])
        label = sample_dict.get("label")
        metadata = sample_dict.get("metadata", {})
        metadata["original_size"] = np.array(image.shape[:2])
        
        if self.transform:
            image, label = self.transform(image, label)
            
        sample = Sample(
            image=image,
            image_id=sample_dict["image_id"],
            image_path=sample_dict["image_path"],
            label=label,
            metadata=metadata,
        )
        
        if not sample.validate():
            raise ValueError(f"样本 {idx} 的格式不正确")
            
        return sample.to_dict()