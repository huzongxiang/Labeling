"""
Florence模型后处理流水线

作者: zongxiang hu
创建日期: 2024-01-07
最后修改: 2024-01-07
"""

from typing import Dict, Any
from src.pipeline.components.base import BasePostPipeline
from src.utils.convert import to_list


class DetectionPostPipeline(BasePostPipeline):
    """
    Florence目标检测后处理流水线
    
    输入格式:
    {
        "<OD>": {
            "bboxes": [[x1, y1, x2, y2], ...],  # 边界框坐标列表
            "labels": ["label1", "label2", ...]  # 标签列表
        }
    }
    
    输出格式:
    {
        "boxes": [[x1, y1, x2, y2], ...],  # 边界框坐标列表
        "class_names": ["label1", "label2", ...]  # 类别名称列表
    }
    """
    
    def format_predictions(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        results = pred["<OD>"]
        return {
            "boxes": to_list(results["bboxes"]),
            "class_names": results["labels"]
        }


class DenseRegionCaptionPostPipeline(BasePostPipeline):
    """
    Florence密集区域描述后处理流水线
    
    输入格式:
    {
        "<DENSE_REGION_CAPTION>": {
            "bboxes": [[x1, y1, x2, y2], ...],  # 边界框坐标列表
            "labels": ["description1", "description2", ...]  # 区域描述列表
        }
    }
    
    输出格式:
    {
        "boxes": [[x1, y1, x2, y2], ...],  # 边界框坐标列表
        "class_names": ["description1", "description2", ...]  # 区域描述列表
    }
    """
    
    def format_predictions(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        results = pred["<DENSE_REGION_CAPTION>"]
        return {
            "boxes": to_list(results["bboxes"]),
            "class_names": results["labels"]
        }


class RegionProposalPostPipeline(BasePostPipeline):
    """
    Florence区域提议后处理流水线
    
    输入格式:
    {
        "<REGION_PROPOSAL>": {
            "bboxes": [[x1, y1, x2, y2], ...],  # 边界框坐标列表
            "labels": ["", "", ...]  # 空标签列表
        }
    }
    
    输出格式:
    {
        "boxes": [[x1, y1, x2, y2], ...],  # 边界框坐标列表
        "class_names": ["region_0", "region_1", ...]  # 区域编号列表
    }
    """
    
    def format_predictions(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        results = pred["<REGION_PROPOSAL>"]
        return {
            "boxes": to_list(results["bboxes"]),
            "class_names": [f"region_{i}" for i in range(len(results["bboxes"]))]
        }


class PhraseGroundingPostPipeline(BasePostPipeline):
    """
    Florence短语定位后处理流水线
    
    输入格式:
    {
        "<CAPTION_TO_PHRASE_GROUNDING>": {
            "bboxes": [[x1, y1, x2, y2], ...],  # 边界框坐标列表
            "labels": ["phrase1", "phrase2", ...]  # 短语列表
        }
    }
    
    输出格式:
    {
        "boxes": [[x1, y1, x2, y2], ...],  # 边界框坐标列表
        "class_names": ["phrase1", "phrase2", ...]  # 短语列表
    }
    """
    
    def format_predictions(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        results = pred["<CAPTION_TO_PHRASE_GROUNDING>"]
        return {
            "boxes": to_list(results["bboxes"]),
            "class_names": results["labels"]
        }


class OpenVocabularyDetectionPostPipeline(BasePostPipeline):
    """
    Florence开放词汇检测后处理流水线
    
    输入格式:
    {
        "<OPEN_VOCABULARY_DETECTION>": {
            "bboxes": [[x1, y1, x2, y2], ...],  # 边界框坐标列表
            "bboxes_labels": ["label1", "label2", ...]  # 标签列表
        }
    }
    
    输出格式:
    {
        "boxes": [[x1, y1, x2, y2], ...],  # 边界框坐标列表
        "class_names": ["label1", "label2", ...]  # 类别名称列表
    }
    """
    
    def format_predictions(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        results = pred["<OPEN_VOCABULARY_DETECTION>"]
        return {
            "boxes": to_list(results["bboxes"]),
            "class_names": results["bboxes_labels"]
        }


class ReferringExpressionPostPipeline(BasePostPipeline):
    """
    Florence指代表达式分割后处理流水线
    
    输入格式:
    {
        "<REFERRING_EXPRESSION_SEGMENTATION>": {
            "polygons": [[[x1, y1], [x2, y2], ...]],  # 多边形点坐标列表
            "labels": [""]  # 标签列表
        }
    }
    
    输出格式:
    {
        "polygons": [[[x1, y1], [x2, y2], ...]],  # 多边形点坐标列表
        "class_names": ["text_input"]  # 指代表达式文本列表
    }
    """
    
    def format_predictions(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        results = pred["<REFERRING_EXPRESSION_SEGMENTATION>"]
        return {
            "polygons": to_list(results["polygons"]),
            "class_names": [label if label else "region" for label in results["labels"]]
        }
        