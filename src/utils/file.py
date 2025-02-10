import json
import shutil
from pathlib import Path
from typing import Any, Dict, Union, Optional, List

def extract_archive(filepath: Path, extract_dir: Optional[Path] = None) -> Path:
    """
    解压文件
    
    Args:
        filepath: 待解压的文件路径
        extract_dir: 解压目标目录，默认为文件所在目录
        
    Returns:
        解压后的目录路径
    """
    extract_dir = extract_dir or filepath.parent / filepath.stem
    if not extract_dir.exists() or not list(extract_dir.iterdir()):
        print(f"Extracting {filepath.name}...")
        shutil.unpack_archive(str(filepath), str(extract_dir))
    return extract_dir

def load_json(filepath: Union[str, Path]) -> Dict:
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """保存数据到JSON文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

class ResultWriter:
    """结果写入器"""
    
    def __init__(self, filepath: Union[str, Path], indent: int = 2):
        """
        初始化结果写入器
        
        Args:
            filepath: 保存路径
            indent: JSON缩进空格数
        """
        self.filepath = Path(filepath)
        self.indent = indent
        self.results = []
        
    def __enter__(self):
        """进入上下文"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载已有结果
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.results = data.get("results", [])
            except:
                self.results = []
                
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时保存结果"""
        if not exc_type:  # 如果没有异常发生
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "results": self.results,
                        "total_images": len(self.results)
                    },
                    f,
                    indent=self.indent,
                    ensure_ascii=False
                )
                
    def write(self, batch_results: List[Dict]) -> None:
        """
        写入一个批次的结果
        
        Args:
            batch_results: 批次结果列表
        """
        self.results.extend(batch_results) 