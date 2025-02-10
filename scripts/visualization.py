"""
可视化工具

作者: zongxiang hu
创建日期: 2024-01-08
最后修改: 2024-01-08
"""

import cv2
import json
import numpy as np
import supervision as sv
from pathlib import Path
from typing import Dict, List, Any


def visualize_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    class_names: List[str],
    scores: List[float] = None,
    masks: np.ndarray = None,
    with_mask: bool = False
) -> np.ndarray:
    """
    使用 supervision 库可视化检测结果
    
    Args:
        image: 原始图像
        boxes: 边界框坐标列表 [[x1, y1, x2, y2], ...]
        class_names: 类别名称列表
        scores: 置信度分数列表（可选）
        masks: 分割掩码（可选）
        with_mask: 是否显示分割掩码
        
    Returns:
        绘制了检测结果的图像
    """
    # 准备检测结果
    class_ids = np.array(list(range(len(class_names))))
    
    # 创建检测对象
    detections = sv.Detections(
        xyxy=np.array(boxes),
        class_id=class_ids,
        mask=masks.astype(bool) if masks is not None else None
    )
    
    # 准备标签
    if scores is not None:
        labels = [
            f"{name} {score:.2f}" 
            for name, score in zip(class_names, scores)
        ]
    else:
        labels = class_names
    
    # 创建标注器
    box_annotator = sv.BoxAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=5
    )
    mask_annotator = sv.MaskAnnotator() if with_mask else None
    
    # 绘制检测框和标签
    annotated_frame = box_annotator.annotate(
        scene=image.copy(), 
        detections=detections,
        labels=labels  # 直接在 BoxAnnotator 中添加标签
    )
    
    # 如果需要，绘制分割掩码
    if with_mask and masks is not None:
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
    
    return annotated_frame


def visualize_results(
    results_file: str, 
    output_dir: str,
    with_mask: bool = False
) -> None:
    """
    可视化标注结果
    
    Args:
        results_file: 结果文件路径
        output_dir: 输出目录
        with_mask: 是否显示分割掩码
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取结果文件
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    # 处理每张图片
    for item in data["results"]:
        # 读取图片
        image_path = item["image_path"]
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            continue
            
        # 获取预测结果
        predictions = item["predictions"]
        boxes = predictions["boxes"]
        class_names = predictions["class_names"]
        scores = predictions.get("scores")  # 可能不存在
        masks = predictions.get("masks")  # 可能不存在
        
        # 如果有检测结果则绘制
        if boxes and class_names:
            # 绘制检测结果
            image = visualize_detections(
                image=image,
                boxes=boxes,
                class_names=class_names,
                scores=scores,
                masks=masks,
                with_mask=with_mask
            )
            
        # 保存结果
        output_path = output_dir / f"{item['image_id']}_vis.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"已保存可视化结果到: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化标注结果")
    parser.add_argument("--results", type=str, required=True, help="结果文件路径")
    parser.add_argument("--output", type=str, default="outputs/visualization", help="输出目录")
    parser.add_argument("--with-mask", action="store_true", help="是否显示分割掩码")
    args = parser.parse_args()
    
    visualize_results(args.results, args.output, args.with_mask)
    