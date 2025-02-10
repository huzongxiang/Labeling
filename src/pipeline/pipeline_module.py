"""
推理流水线主入口

作者: zongxiang hu
创建日期: 2024-01-03
最后修改: 2024-01-03

该模块实现了推理流水线的主要逻辑,包括:
- 数据加载和预处理
- 模型推理和后处理
- 结果保存
"""

from typing import Dict, Optional
from src.data.data_module import DataModule
from src.models.labeling_module import LabelingModule
from src.pipeline.components.base import BasePrePipeline, BasePostPipeline
from src.utils.logger import setup_logger

class PipelineModule:
    """推理流水线"""
    
    def __init__(
        self,
        data_module: DataModule,
        labeling_module: LabelingModule,
        prepipeline: Optional[BasePrePipeline] = None,
        postpipeline: Optional[BasePostPipeline] = None,
    ):
        """
        初始化流水线
        
        Args:
            data_module: 数据处理模块
            labeling_module: 标注模块
            prepipeline: 预处理组件
            postpipeline: 后处理组件
            config: 配置信息
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.data_module = data_module
        self.labeling_module = labeling_module
        self.prepipeline = prepipeline   
        self.postpipeline = postpipeline
        
        self.logger.info("初始化推理流水线...")
        
    def run(self) -> None:
        """执行流水线"""
        try:
            # 1. 获取原始数据
            self.logger.info("开始获取数据...")
            data = self.data_module.process()
            
            # 2. 设置数据集和加载器
            self.logger.info("设置数据集...")
            dataloader = self.prepipeline(data)
            
            # 3. 执行推理
            self.logger.info("开始执行自动标注...")
            
            for batch in dataloader:
                # 模型预测
                outputs = self.labeling_module(batch)
                
                # 后处理
                self.postpipeline(outputs)
            
            self.logger.info("流水线执行完成")
            
        except Exception as e:
            self.logger.error(f"流水线执行出错: {str(e)}")
            raise e
