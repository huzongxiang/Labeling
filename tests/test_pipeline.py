import pytest
from pathlib import Path
from omegaconf import OmegaConf
from src.pipeline import Pipeline
from src.data.data_module import DataModule
from src.models.labeling_module import LabelingModule

@pytest.fixture
def mock_config():
    """创建测试配置"""
    return OmegaConf.create({
        "config": {
            "save_format": "json",
            "device": "cpu"
        }
    })

@pytest.fixture
def mock_data_module(mocker):
    """模拟数据模块"""
    data_module = mocker.Mock(spec=DataModule)
    data_module.get_dataloader.return_value = [(torch.randn(1, 3, 224, 224), {})]
    return data_module

@pytest.fixture
def mock_labeling_module(mocker):
    """模拟标注模块"""
    labeling_module = mocker.Mock(spec=LabelingModule)
    labeling_module.predict.return_value = [{"boxes": [], "labels": []}]
    return labeling_module

def test_pipeline_initialization(mock_config, mock_data_module, mock_labeling_module):
    """测试流水线初始化"""
    pipeline = Pipeline(
        data_module=mock_data_module,
        labeling_module=mock_labeling_module,
        config=mock_config
    )
    assert pipeline.data_module == mock_data_module
    assert pipeline.labeling_module == mock_labeling_module
    assert pipeline.config == mock_config

def test_pipeline_run(mock_config, mock_data_module, mock_labeling_module):
    """测试流水线运行"""
    pipeline = Pipeline(
        data_module=mock_data_module,
        labeling_module=mock_labeling_module,
        config=mock_config
    )
    results = pipeline.run()
    assert isinstance(results, dict)
    assert "results" in results 