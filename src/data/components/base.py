from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from src.utils.data import DataInfo
from src.utils.file import save_json
from src.utils.logger import setup_logger

class BaseSource(ABC):
    """数据源的基类"""
    
    def __init__(self, name: str, data_dir: Union[str, Path]):
        """
        初始化数据源
        
        Args:
            name: 数据源名称，用于创建对应的目录
            data_dir: 数据根目录，由配置文件控制
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.name = name
        self.data_dir = Path(data_dir)
        
        # 固定的子目录结构
        self.download_dir = self.data_dir / "downloads" / name
        self.raw_dir = self.data_dir / "raw" / name
        
        # 创建目录
        for dir_path in [self.download_dir, self.raw_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"创建目录: {dir_path}")
        
        self.logger.info(f"初始化数据源: {name}")
    
    def prepare_structure(self) -> Tuple[Path, Path]:
        """
        准备标准的数据目录结构
        
        Returns:
            (images_dir, annotations_dir) 元组
        """
        images_dir = self.raw_dir / "images"
        annotations_dir = self.raw_dir / "annotations"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        return images_dir, annotations_dir
    
    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """
        获取数据，返回统一格式：
        {
            "images_dir": Path,      # 图片目录
            "annotations_dir": Path,  # 标注目录
            "annotations": Optional[Dict]  # 标注数据（如果有）
        }
        """
        pass
    
    @abstractmethod
    def get_info(self) -> DataInfo:
        """获取数据源信息"""
        pass
    
    def validate(self) -> bool:
        """验证数据源的有效性"""
        return True
    
    def clean(self) -> None:
        """清理临时文件"""
        pass

class BaseProcessor(ABC):
    """数据处理基类"""
    
    def __init__(
        self,
        dataset_name: str,
        data_dir: Union[str, Path],
        save_format: str = "jpg"
    ):
        self.logger = setup_logger(self.__class__.__name__)
        
        # 使用配置的根目录
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.save_format = save_format
        
        # 处理后的数据目录
        self.processed_dir = self.data_dir / "processed" / dataset_name
        self.processed_images_dir = self.processed_dir / "images"  # 仅在需要修改图片时使用
        self.processed_annotations_dir = self.processed_dir / "annotations"
        
        # 创建目录
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.processed_annotations_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"初始化处理器: {dataset_name}")
    
    def save_data_info(
        self,
        images_dir: Union[str, Path],
        annotations_file: Union[str, Path],
        modified: bool = None,
    ) -> Path:
        """
        保存数据信息到data.json
        
        Args:
            images_dir: 图片目录路径
            annotations_file: 标注文件的路径
            modified: 是否修改了图片
        
        Returns:
            data.json的路径
        """
        # data.json结构:
        # {
        #   "images_dir": "/path/to/images",
        #   "annotations_file": "/path/to/annotations/annotations.json",
        #   "modified": true/false
        # }
        data_info = {
            "images_dir": images_dir.as_posix() if isinstance(images_dir, Path) else images_dir,
            "annotations_file": annotations_file.as_posix() if isinstance(annotations_file, Path) else annotations_file,
            "modified": modified
        }
        
        data_file = self.processed_dir / "data.json"
        save_json(data_info, data_file)
        return data_file
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理数据
        
        Args:
            data: 包含原始数据路径的字典
                {
                    "images_dir": Path,      # 原始图片目录
                    "annotations_dir": Path,  # 原始标注目录
                    "annotations": Optional[Dict]  # 原始标注数据
                }
                
        Returns:
            处理结果字典:
                {
                    "images_dir": Path,      # 图片目录（可能是原始目录或处理后目录）
                    "annotations_dir": Path,  # 处理后的标注目录
                    "annotations": Dict,      # 处理后的标注数据
                    "data_file": Path        # data.json的路径
                }
        """
        pass
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """验证输入数据的有效性"""
        required_keys = ["images_dir", "annotations_dir"]
        return all(key in data for key in required_keys)
    
    def get_output_info(self) -> Dict:
        """获取输出数据的信息"""
        return {}
    
    def cleanup(self) -> None:
        """清理临时文件"""
        pass
