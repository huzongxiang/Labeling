"""
推理流水线组件

包含:
- 预处理组件: ImagePrePipeline
- 后处理组件: YOLOPostPipeline, GroundingDINOPostPipeline, SAMPostPipeline
"""

from .pipeline import (
    ImagePrePipeline,
    YOLOPostPipeline,
    GroundingDINOPostPipeline,
    SAMPostPipeline,
    DefaultPrePipeline,
    DefaultPostPipeline,
)
from .florence import (
    DetectionPostPipeline,
    DenseRegionCaptionPostPipeline,
    RegionProposalPostPipeline,
    PhraseGroundingPostPipeline,
    ReferringExpressionPostPipeline,
    OpenVocabularyDetectionPostPipeline,
)

__all__ = [
    # 基类
    "DefaultPrePipeline",
    "DefaultPostPipeline",
    # 具体实现
    "ImagePrePipeline",
    "YOLOPostPipeline", 
    "GroundingDINOPostPipeline",
    "SAMPostPipeline",
    # Florence
    "DetectionPostPipeline",
    "DenseRegionCaptionPostPipeline",
    "RegionProposalPostPipeline",
    "PhraseGroundingPostPipeline",
    "ReferringExpressionPostPipeline",
    "OpenVocabularyDetectionPostPipeline",
]
