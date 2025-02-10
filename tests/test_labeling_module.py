import pytest
import torch
from src.models.labeling_module import LabelingModule
from src.models.components import BaseModel
from src.models.components.postprocessor.base import BasePostProcessor

class MockModel(BaseModel):
    """模拟模型"""
    def __call__(self, images):
        return [{"boxes": torch.tensor([[0, 0, 1, 1]]), "scores": torch.tensor([0.9])}]

class MockPostProcessor(BasePostProcessor):
    """模拟后处理器"""
    def process(self, outputs):
        return outputs

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def mock_postprocessor_cls():
    return MockPostProcessor

def test_labeling_module_initialization(mock_model, mock_postprocessor_cls):
    """测试标注模块初始化"""
    module = LabelingModule(
        model=mock_model,
        postprocessor_cls=mock_postprocessor_cls,
        config={"device": "cpu"}
    )
    assert module.model == mock_model
    assert isinstance(module.postprocessor, mock_postprocessor_cls)

def test_labeling_module_predict(mock_model, mock_postprocessor_cls):
    """测试标注模块预测"""
    module = LabelingModule(
        model=mock_model,
        postprocessor_cls=mock_postprocessor_cls,
        config={"device": "cpu"}
    )
    images = [torch.randn(3, 224, 224)]
    results = module.predict(images)
    assert isinstance(results, list)
    assert len(results) == 1
    assert "boxes" in results[0] 