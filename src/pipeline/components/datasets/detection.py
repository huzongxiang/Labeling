from pathlib import Path
from typing import Dict, Any, Optional, Union
import torchvision.transforms as T
from .base import BaseDataset
from src.utils.file import load_json
from src.utils.data import Sample

class DetectionDataset(BaseDataset):
    """目标检测数据集"""
    
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
                sample["boxes"] = annotations[img_path.stem]["boxes"]
                sample["labels"] = annotations[img_path.stem]["labels"]
            self.samples.append(sample)
        
        self.logger.info(f"加载了 {len(self.samples)} 个图像，其中 {len(annotations)} 个有标注")
        
    def __getitem__(self, idx: int) -> Sample:
        """获取单个样本"""
        sample_dict = self.samples[idx]
        image = self._load_image(sample_dict["image_path"])
        
        if self.transform:
            image = self.transform(image)
            
        label = None
        if "boxes" in sample_dict:
            label = {
                "boxes": sample_dict["boxes"],
                "labels": sample_dict["labels"]
            }
            
        sample = Sample(
            image=image,
            image_id=sample_dict["image_id"],
            image_path=sample_dict["image_path"],
            label=label,
            metadata=sample_dict.get("metadata")
        )
        
        if not sample.validate():
            raise ValueError(f"样本 {idx} 的格式不正确")
            
        return sample 