# 自动标注工具

基于深度学习的多模型自动标注工具，支持目标检测、图像分割等多种任务，提供灵活配置和高效推理能力。

## 主要特性

- **多模型支持**：集成YOLO、DINO、SAM、SAHI、Florence等前沿模型
- **多数据格式**：支持图像(JPG/PNG)、视频(MP4/AVI)、COCO数据集
- **模块化设计**：数据、模型、流水线模块解耦，易于扩展
- **高效推理**：支持GPU加速、混合精度、批处理等优化策略
- **灵活配置**：通过Hydra实现层级化配置管理

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8 (GPU用户)
- [aria2](https://aria2.github.io/) (推荐用于模型下载)

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/yourusername/auto-labeling-tool.git
cd auto-labeling-tool

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 设置项目根目录（可选）
echo "PROJECT_ROOT=$(pwd)" >> .env
```

### 基础用法

```bash
# 使用默认配置运行（YOLO模型 + 视频数据）
python src/labeling.py

# 指定模型和数据源
python src/labeling.py model=yolo data=image
python src/labeling.py model=sam data=coco

# 自定义输出目录
python src/labeling.py paths.output_dir=./custom_output
```

## 进阶配置

### 模型配置示例

`configs/model/yolo.yaml`:
```yaml
model:
  _target_: src.models.components.YOLODetector
  model_name: "yolov8n.pt"  # 可选模型: yolov8n/s/m/l/x
  device: "cuda"  # 切换为cuda使用GPU
  conf_thres: 0.25  # 置信度阈值
  iou_thres: 0.45   # IoU阈值
  classes: [0]      # 指定检测类别（COCO类别ID）
```

### 流水线配置

`configs/pipeline/pipeline.yaml`:
```yaml
prepipeline:
  dataset_cls:
    transform:
      _target_: src.pipeline.components.datasets.transform.YOLOTransform
      target_size: 640  # 输入尺寸
  batch_size: 16        # 批处理大小

postpipeline:
  save_dir: ${paths.output_dir}/annotations
  save_name: results.json
```

### 常用命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| model | 选择模型 | model=yolo/sam/dino |
| data | 数据格式 | data=image/video/coco |
| seed | 随机种子 | seed=42 |
| paths.output_dir | 输出目录 | paths.output_dir=./results |

## 模型库

| 模型       | 类型       | 配置文件            | 特点                      |
|------------|------------|---------------------|---------------------------|
| YOLOv8     | 目标检测   | `model/yolo.yaml`    | 实时检测，轻量级          |
| SAM        | 图像分割   | `model/sam.yaml`     | 零样本分割                |
| SAHI       | 小目标检测 | `model/sahi.yaml`   | 切片推理提升小目标检测    |
| Florence   | 多任务模型 | `model/florence.yaml`| 支持检测/描述/定位等多任务|

## 目录结构

```
.
├── configs/                  # 配置文件
│   ├── hydra/                # Hydra默认配置
│   ├── labeling.yaml         # 主配置文件
│   ├── model/                # 模型配置
│   │   ├── yolo.yaml         # YOLO配置
│   │   ├── sam.yaml          # SAM配置
│   │   └── ...               # 其他模型
│   ├── paths/                # 路径配置
│   └── pipeline/            # 流水线配置
├── src/                      # 源代码
│   ├── data/                 # 数据模块
│   ├── models/               # 模型实现
│   ├── pipeline/             # 流水线逻辑
│   └── utils/               # 工具函数
├── outputs/                  # 默认输出目录
├── requirements.txt         # 依赖列表
└── README.md                # 项目文档
```

## 常见问题

### 模型下载问题
```bash
# 设置HF镜像(中国大陆用户推荐)
export HF_ENDPOINT=https://hf-mirror.com
```

### 多GPU支持
```yaml
# 在模型配置中添加
device: "cuda"
gpu_ids: [0, 1]  # 指定使用的GPU
```

### 性能优化
```yaml
# model/sam.yaml
dtype: "bfloat16"  # 使用混合精度
batch_size: 4       # 增大批处理大小
```

## 贡献指南

欢迎通过Issue和PR参与项目开发，请遵循以下规范：
1. 新功能开发请创建特性分支 (feat/xxx)
2. Bug修复请基于hotfix分支
3. 提交前执行代码格式化：
```bash
black . && isort .
```

## 许可证

本项目采用 [MIT License](LICENSE)，保留署名权利。

Copyright (c) 2024 Zongxiang Hu
