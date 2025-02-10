"""
数据处理模块

作者: zongxiang hu
创建日期: 2024-01-03
最后修改: 2024-01-03

该模块实现了数据处理模块的主要逻辑,包括:
- 数据加载和预处理
- 数据验证
- 数据处理
"""

from typing import Optional, Dict, Any
from pathlib import Path
from src.utils.logger import setup_logger
from src.data.components.base import BaseSource, BaseProcessor

class DataModule:
    """数据处理模块"""
    
    def __init__(
        self,
        source: BaseSource,
        processor: Optional[BaseProcessor] = None,
        config: Dict = None,
    ):
        """
        初始化数据处理模块
        
        Args:
            source: 数据源
            processor: 数据处理器（可选）
            config: 配置信息
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("初始化数据处理模块...")
        
        self.config = config or {}
        self.source = source
        self.processor = processor
    
    def process(self) -> Dict[str, Any]:
        """
        执行数据处理流程
        
        Returns:
            处理后的数据信息
        """
        try:
            # 验证数据源
            self.logger.info("开始验证数据源...")
            if not self.source.validate():
                error_msg = f"数据源验证失败: {self.source.get_info().name}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 获取数据
            self.logger.info("开始获取数据...")
            data = self.source.get_data()
            
            # 验证数据格式
            self.logger.info("验证数据格式...")
            if not self._validate_data_format(data):
                error_msg = "数据格式不正确"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 处理数据
            if self.processor:
                self.logger.info(f"使用 {self.processor.__class__.__name__} 处理数据...")
                data = self.processor.process(data)
                if not self._validate_data_format(data):
                    error_msg = "处理后数据格式不正确"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
            
            self.logger.info("数据处理完成")
            return data
            
        except Exception as e:
            self.logger.error(f"数据处理出错: {str(e)}")
            self.cleanup()
            raise e
    
    def _validate_data_format(self, data: Dict) -> bool:
        """验证数据格式是否符合规范"""
        required_keys = ["images_dir", "annotations_dir"]
        if not all(key in data for key in required_keys):
            self.logger.warning(f"缺少必要的键: {required_keys}")
            return False
        if not isinstance(data["images_dir"], Path):
            self.logger.warning("images_dir 不是 Path 类型")
            return False
        if not isinstance(data["annotations_dir"], Path):
            self.logger.warning("annotations_dir 不是 Path 类型")
            return False
        return True
    
    def cleanup(self) -> None:
        """清理临时资源"""
        self.logger.info("开始清理临时资源...")
        if self.processor:
            self.processor.cleanup()
        self.source.clean() 