import pytest
from pathlib import Path
import torch
from src.data.data_module import DataModule
from src.data.components.base import BaseSource, BaseProcessor
from torch.utils.data import Dataset

class MockSource(BaseSource):
    """模拟数据源"""
    def get_data(self):
        return {
            "images_dir": Path("tests/data/images"),
            "annotations_dir": Path("tests/data/annotations")
        }
    
    def validate(self):
        return True

class MockDataset(Dataset):
    """模拟数据集"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        return torch.randn(3, 224, 224), {}

@pytest.fixture
def mock_source():
    return MockSource()

@pytest.fixture
def mock_dataset_cls():
    return MockDataset

def test_data_module_initialization(mock_source, mock_dataset_cls):
    """测试数据模块初始化"""
    data_module = DataModule(
        source=mock_source,
        dataset_cls=mock_dataset_cls,
        config={"dataloader": {"batch_size": 1}}
    )
    assert data_module.source == mock_source
    assert data_module.dataset_cls == mock_dataset_cls

def test_data_module_setup(mock_source, mock_dataset_cls):
    """测试数据模块设置"""
    data_module = DataModule(
        source=mock_source,
        dataset_cls=mock_dataset_cls,
        config={"dataloader": {"batch_size": 1}}
    )
    data_module.setup()
    assert data_module.dataset is not None
    assert data_module.dataloader is not None 