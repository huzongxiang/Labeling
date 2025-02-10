"""
Florence模型封装

作者: zongxiang hu
创建日期: 2024-01-07
最后修改: 2024-01-07
"""

import torch
from typing import Dict, List, Optional, Union, Any
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from src.models.components.base import BaseModel
from src.utils.device import device_context
from src.utils.logger import setup_logger


class FlorenceModel(BaseModel):
    """
    Florence模型封装
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
    """
    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        task_prompt: str = "<OD>",  # 任务提示词
        text_input: Optional[str] = None,  # 文本输入（部分任务需要）
        max_new_tokens: int = 1024,
        num_beams: int = 3
    ):
        """
        初始化Florence模型
        
        Args:
            model_name: HuggingFace模型名称或本地路径
            device: 运行设备
            cache_dir: 权重缓存目录，默认使用HF_HOME
            task_prompt: 任务提示词，如 "<OD>", "<DENSE_REGION_CAPTION>" 等
            text_input: 文本输入，部分任务需要
            max_new_tokens: 生成最大token数
            num_beams: beam search数量
        """
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(f"初始化Florence模型: {model_name}")
        
        self.task_prompt = task_prompt
        self.text_input = text_input
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        
        # 加载模型和处理器
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
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
            
        # 构建完整提示词
        prompt = self.task_prompt
        if self.text_input is not None:
            prompt = self.task_prompt + self.text_input
            
        # 准备输入
        inputs = self.processor(
            text=[prompt] * batch_size,  # 为每个图像复制提示词
            images=batch['image'],
            return_tensors="pt"
        ).to(self.device, torch.float16)
        
        return inputs
        
    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, List]]
    ) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Args:
            batch: DataLoader 返回的批次数据，包含：
                - image: torch.Tensor 或 List[PIL.Image]，批次图像数据
                - image_id: List[str]，图像ID列表
            
        Returns:
            预测结果列表，每个元素为一张图片的预测结果
        """
        # 预处理
        inputs = self.preprocess(batch)
        
        # 打印维度信息用于调试
        self.logger.info(f"input_ids shape: {inputs['input_ids'].shape}")
        self.logger.info(f"pixel_values shape: {inputs['pixel_values'].shape}")
        
        # 推理
        with device_context(self.device):
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                early_stopping=False,
                do_sample=False,
                num_beams=self.num_beams
            )
            
        # 解码生成的文本
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )

        # 后处理每个样本
        results = []
        for i, text in enumerate(generated_text):
            results.append(self.postprocess(text, {
                'image': [batch['image'][i]],
                'image_id': [batch['image_id'][i]]
            }))

        return results
        
    def postprocess(
        self,
        outputs: str,
        datainfo: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        后处理Florence输出
        
        Args:
            outputs: 生成的文本
            metadata: 批次元数据，包含原始图像信息
            
        Returns:
            处理后的预测结果，直接返回模型的解析结果
        """
        # 获取图像尺寸
        image = datainfo['image'][0]  # 当前只支持批次大小为1
        if isinstance(image, Image.Image):
            image_size = image.size
        else:
            image_size = image.shape[:2][::-1]  # HW -> WH

        # 解析生成的文本并直接返回结果
        return self.processor.post_process_generation(
            outputs,
            task=self.task_prompt,
            image_size=image_size
        )
