import pytest
import torch
import os
from pathlib import Path

@pytest.fixture(autouse=True)
def setup_test_env():
    """设置测试环境"""
    # 设置环境变量
    os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.parent)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # 确保使用CPU进行测试
    torch.set_grad_enabled(False)
    
    # 创建测试数据目录
    test_data_dir = Path("tests/data")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    (test_data_dir / "images").mkdir(exist_ok=True)
    (test_data_dir / "annotations").mkdir(exist_ok=True)
    
    yield
    
    # 清理测试数据
    import shutil
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir) 