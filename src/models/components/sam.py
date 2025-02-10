import torch
import numpy as np
from typing import Optional, Tuple, Dict, Union, List
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.models.components.base import BaseModel
from src.utils.device import device_context

class SAM2Model(BaseModel):
    """SAM2模型封装"""
    
    def __init__(
        self,
        model_name: str = "facebook/sam2-hiera-large",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        multimask_output: bool = False
    ):
        """
        初始化SAM2模型
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备
            dtype: 数据类型
            cache_dir: 缓存目录
            multimask_output: 是否输出多个掩码
        """
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)
        self.dtype = dtype
        self.multimask_output = multimask_output
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_path)
        
    def preprocess(
        self, 
        batch: Dict[str, Union[torch.Tensor, List]]
    ) -> List[Dict]:
        """预处理输入"""
        return batch
        
    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, List]]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        批量预测分割掩码
        
        Args:
            batch: DataLoader 返回的批次数据，包含：
                - image: torch.Tensor 或 List[PIL.Image]，批次图像数据
                - image_id: List[str]，图像ID列表
                - label: Optional[Dict]，可能包含:
                    - boxes: torch.Tensor，边界框坐标
                    - points: torch.Tensor，点坐标
                    - point_labels: torch.Tensor，点标签
            
        Returns:
            List of (masks, scores, logits)，每个元素对应一张图片的分割结果
        """
        batch = self.preprocess(batch)
        results = []
        
        with device_context(self.device, self.dtype):
            for i in range(len(batch['image_id'])):
                image = batch['image'][i]
                label = batch.get('label', None)
                
                # 获取标注数据（如果有）
                boxes = label[i].get('boxes', None) if label is not None else None
                points = label[i].get('points', None) if label is not None else None
                point_labels = label[i].get('point_labels', None) if label is not None else None
                
                # 设置图像并预测
                self.predictor.set_image(image)
                result = self.predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    box=boxes,
                    multimask_output=self.multimask_output
                )
                results.append(result)
                
        return self.postprocess(results)
    
    def postprocess(
        self, 
        results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> List[Dict]:
        """后处理模型输出"""
        return results
    