"""
可视化标注结果

作者: zongxiang hu
创建日期: 2024-01-08
最后修改: 2024-01-08
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# 定义颜色映射
COLORS = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8).tolist()


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    class_names: List[str],
    scores: List[float] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制边界框和标签
    
    Args:
        image: 原始图像
        boxes: 边界框坐标列表 [[x1, y1, x2, y2], ...]
        class_names: 类别名称列表
        scores: 置信度分数列表（可选）
        thickness: 线条粗细
        
    Returns:
        绘制了边界框的图像
    """
    image = image.copy()
    
    for i, (box, label) in enumerate(zip(boxes, class_names)):
        # 获取框的坐标
        x1, y1, x2, y2 = map(int, box)
        
        # 获取颜色
        color = COLORS[i % len(COLORS)]
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 准备标签文本
        if scores is not None:
            text = f"{label}: {scores[i]:.2f}"
        else:
            text = label
            
        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness
        )
        
        # 绘制标签背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            image,
            text,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness
        )
        
    return image


def visualize_results(results_file: str, output_dir: str) -> None:
    """
    可视化标注结果
    
    Args:
        results_file: 结果文件路径
        output_dir: 输出目录
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
        
        # 如果有检测结果则绘制
        if boxes and class_names:
            # 绘制边界框和标签
            image = draw_boxes(image, boxes, class_names, scores)
            
        # 保存结果
        output_path = output_dir / f"{item['image_id']}_vis.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"已保存可视化结果到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化标注结果")
    parser.add_argument("--results", type=str, required=True, help="结果文件路径")
    parser.add_argument("--output", type=str, default="outputs/visualization", help="输出目录")
    args = parser.parse_args()
    
    visualize_results(args.results, args.output)


if __name__ == "__main__":
    main() 