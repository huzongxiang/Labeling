from typing import Dict, Any, Union, Optional, Tuple
from pathlib import Path
import cv2
from tqdm import tqdm
from src.data.components.base import BaseProcessor
from src.utils.file import save_json
from src.utils.logger import setup_logger

class DefaultProcessor(BaseProcessor):
    """默认处理器，仅保存data.json"""
    def __init__(
        self,
        dataset_name: str,
        data_dir: Union[str, Path],
        save_format: str = "jpg"
    ):
        super().__init__(dataset_name, data_dir, save_format)
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("初始化简单处理器")
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """直接保存data.json，不做任何处理"""

        # 构建图片信息并保存data.json
        # data.json结构:
        # {
        #   "images_dir": "/path/to/images",
        #   "annotations_file": "/path/to/annotations/annotations.json"
        # }
        # 获取图片目录和标注目录
        images_dir = data["images_dir"]
        annotations_dir = data["annotations_dir"]
        
        # 保存data.json
        data_file = self.save_data_info(
            images_dir=str(images_dir),
            annotations_file=str(annotations_dir / "annotations.json")
        )
        
        return {
            "images_dir": images_dir,
            "annotations_dir": data["annotations_dir"],
            "annotations": data["annotations"],
            "data_file": data_file
        }


class COCOProcessor(BaseProcessor):
    """COCO数据集处理器"""
    def __init__(
        self,
        dataset_name: str,
        data_dir: Union[str, Path],
        category_id: int = 1,  # 1 for person
        save_format: str = "jpg"
    ):
        super().__init__(dataset_name, data_dir, save_format)
        self.logger = setup_logger(self.__class__.__name__)
        self.category_id = category_id
        self.logger.info(f"初始化COCO处理器，类别ID: {self.category_id}")
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """过滤指定类别并保存标注"""
        # 过滤标注
        annotations = data["annotations"]
        filtered_annotations = {}
        filtered_images = {}
        
        # 收集符合条件的标注和对应的图片信息
        for anno in annotations["annotations"]:
            if anno["category_id"] == self.category_id:
                image_id = str(anno["image_id"]).zfill(12)
                if image_id not in filtered_annotations:
                    filtered_annotations[image_id] = []
                    # 找到对应的图片信息
                    for img in annotations["images"]:
                        if str(img["id"]).zfill(12) == image_id:
                            filtered_images[image_id] = img
                            break
                filtered_annotations[image_id].append(anno)
        
        # 构建新的COCO格式标注
        processed_annotations = {
            "images": list(filtered_images.values()),
            "annotations": [
                anno
                for image_annos in filtered_annotations.values()
                for anno in image_annos
            ],
            "categories": annotations["categories"]
        }
        
        # 保存处理后的标注
        annotations_file = self.processed_annotations_dir / "annotations.json"
        save_json(processed_annotations, annotations_file)
        
        # 构建图片信息并保存data.json
        # data.json结构:
        # {
        #   "images_dir": "/path/to/images",
        #   "annotations_file": "/path/to/annotations/annotations.json"
        # }
        data_file = self.save_data_info(
            images_dir=data["images_dir"],
            annotations_file=annotations_file
        )
        
        return {
            "images_dir": data["images_dir"],  # 使用原始图片目录
            "annotations_dir": self.processed_annotations_dir,
            "annotations": processed_annotations,
            "data_file": data_file
        }

class VideoFrameProcessor(BaseProcessor):
    """视频帧处理器"""
    def __init__(
        self,
        dataset_name: str,
        data_dir: Union[str, Path],
        resize: Optional[Tuple[int, int]] = None,
        save_format: str = "jpg"
    ):
        super().__init__(dataset_name, data_dir, save_format)
        self.logger = setup_logger(self.__class__.__name__)
        self.resize = resize  # (width, height) or None
        
        # 如果需要调整大小，创建processed_images_dir
        if self.resize:
            self.processed_images_dir.mkdir(parents=True, exist_ok=True)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理并保存视频帧"""
        output_dir = self.processed_images_dir if self.resize else data["images_dir"]
        
        # 处理所有帧
        for frame_path in tqdm(list(data["images_dir"].glob(f"*.{self.save_format}")), desc="Processing frames"):
            frame = cv2.imread(str(frame_path))
            output_path = self.processed_images_dir / frame_path.name
            if self.resize:
                # 需要调整大小，读取并处理图片
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    frame = cv2.resize(frame, self.resize)
            cv2.imwrite(str(output_path), frame)
        
        # 创建空的annotations.json
        annotations_file = self.processed_annotations_dir / "annotations.json"
        save_json({}, annotations_file)
        
        # 构建图片信息并保存data.json
        # data.json结构:
        # {
        #   "images_dir": "/path/to/images",
        #   "annotations_file": "/path/to/annotations/annotations.json"
        # }
        data_file = self.save_data_info(output_dir, annotations_file, modified=bool(self.resize))
        
        return {
            "images_dir": output_dir,
            "annotations_dir": self.processed_annotations_dir,
            "annotations": None,
            "data_file": data_file
        } 