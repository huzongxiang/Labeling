import subprocess
from pathlib import Path
from typing import Optional

def download_with_aria2(url: str, save_dir: Path, filename: Optional[str] = None) -> Path:
    """
    使用aria2下载文件
    
    Args:
        url: 下载链接
        save_dir: 保存目录
        filename: 保存的文件名，如果为None则使用url中的文件名
        
    Returns:
        下载文件的路径
    """
    filename = filename or url.split("/")[-1]
    filepath = save_dir / filename
    
    if not filepath.exists():
        print(f"Downloading {filename} with aria2...")
        save_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["aria2c", "-x", "16", "-s", "16", "-d", str(save_dir), "-o", filename, url], 
            check=True
        )
    return filepath 