from pathlib import Path
from typing import Optional, Dict, List, Union
import cv2
from tqdm import tqdm
import shutil
from src.utils.logger import setup_logger

from src.data.components.base import BaseSource, DataInfo
from src.utils.download import download_with_aria2
from src.utils.file import extract_archive, load_json

class URLSource(BaseSource):
    """URL数据源"""
    def __init__(self, url: str, data_dir: str, is_image: bool = True):
        name = Path(url).stem
        super().__init__(name, data_dir)
        self.url = url
        self.is_image = is_image
        
    def get_data(self) -> Dict:
        # 下载到downloads子目录
        file_path = download_with_aria2(self.url, self.download_dir)
        
        # 准备标准目录结构
        images_dir, annotations_dir = self.prepare_structure()
        
        if self.is_image:
            dst_path = images_dir / file_path.name
            shutil.copy2(file_path, dst_path)
        
        return {
            "images_dir": images_dir,
            "annotations_dir": annotations_dir,
            "annotations": None
        }
        
    def get_info(self) -> DataInfo:
        return DataInfo(
            name=Path(self.url).name,
            type="url",
            format=Path(self.url).suffix[1:],
            metadata={"url": self.url}
        )

class ImageSource(BaseSource):
    """本地图像数据源"""
    def __init__(
        self,
        src_dir: Union[str, Path],
        data_dir: str,
        name: str = "images",
        extensions: List[str] = [".jpg", ".jpeg", ".png"],
        recursive: bool = False
    ):
        """
        初始化图像数据源
        
        Args:
            src_dir: 源图像目录
            data_dir: 数据根目录
            name: 数据集名称
            extensions: 支持的图像扩展名
            recursive: 是否递归搜索子目录
        """
        super().__init__(name, data_dir)
        self.src_dir = Path(src_dir)
        self.extensions = extensions
        self.recursive = recursive
        
    def get_data(self) -> Dict:
        """组织图像数据"""
        # 准备标准目录结构
        images_dir, annotations_dir = self.prepare_structure()
        
        # 收集并复制图片
        pattern = "**/*" if self.recursive else "*"
        for ext in self.extensions:
            for src_path in self.src_dir.glob(f"{pattern}{ext}"):
                rel_path = src_path.relative_to(self.src_dir)
                dst_path = images_dir / rel_path
                # 目录已在prepare_structure中创建
                shutil.copy2(src_path, dst_path)
                
        self.logger.info(f"从 {self.src_dir} 复制图像到 {images_dir}")
                
        return {
            "images_dir": images_dir,
            "annotations_dir": annotations_dir,
            "annotations": None
        }
        
    def get_info(self) -> DataInfo:
        return DataInfo(
            name=self.name,
            type="images",
            format="directory",
            size=len(list(self.src_dir.glob("*")))
        )

class VideoSource(BaseSource):
    """视频数据源"""
    def __init__(
        self,
        video_path: str,
        data_dir: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ):
        name = Path(video_path).stem
        super().__init__(name, data_dir)
        # 视频文件应该放在downloads子目录下
        self.video_path = self.download_dir / video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        
    def get_data(self) -> Dict:
        # 准备标准目录结构
        images_dir, annotations_dir = self.prepare_structure()
        
        # 提取帧到raw子目录
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.end_frame is None:
            self.end_frame = total_frames
            
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            
            for frame_id in tqdm(
                range(self.start_frame, self.end_frame, self.step),
                desc=f"Extracting frames from {self.video_path.name}"
            ):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_path = images_dir / f"{self.video_path.stem}_{frame_id:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                for _ in range(self.step - 1):
                    cap.read()
                    
        finally:
            cap.release()
        
        return {
            "images_dir": images_dir,
            "annotations_dir": annotations_dir,
            "annotations": None
        }
        
    def get_info(self) -> DataInfo:
        return DataInfo(
            name=self.video_path.name,
            type="video",
            format=self.video_path.suffix[1:],
            metadata={"video_path": str(self.video_path)}
        )

class COCOSource(BaseSource):
    """COCO数据集源"""
    def __init__(
        self, 
        images_url: str,
        annotations_url: str,
        data_dir: str
    ):
        super().__init__("coco", data_dir)
        self.images_url = images_url
        self.annotations_url = annotations_url
        
    def get_data(self) -> Dict:
        if not self.validate():
            # 下载到downloads子目录
            images_path = download_with_aria2(self.images_url, self.download_dir)
            annotations_path = download_with_aria2(self.annotations_url, self.download_dir)
            
            # 解压到raw子目录
            extract_archive(images_path, self.raw_dir)
            extract_archive(annotations_path, self.raw_dir)
        
        return {
            "images_dir": self.raw_dir / "train2017",
            "annotations_dir": self.raw_dir / "annotations",
            "annotations": self.load_annotations()
        }
    
    def get_info(self) -> DataInfo:
        return DataInfo(
            name="COCO",
            type="dataset",
            format="zip",
            metadata={
                "images_url": self.images_url,
                "annotations_url": self.annotations_url
            }
        )
    
    def validate(self) -> bool:
        """验证数据集完整性"""
        required_files = [
            self.raw_dir / "train2017",
            self.raw_dir / "annotations" / "instances_train2017.json"
        ]
        return all(f.exists() for f in required_files)
        
    def load_annotations(self) -> Dict:
        """加载COCO标注文件"""
        annotations_file = self.raw_dir / "annotations" / "instances_train2017.json"
        return load_json(annotations_file) 