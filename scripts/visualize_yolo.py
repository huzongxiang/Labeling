import os
import cv2
import numpy as np
from tqdm import tqdm

# 常量定义
DATASET_DIR = "./datas"  # YOLO数据集根目录
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train/images")
TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, "train/labels")
VAL_IMAGES_DIR = os.path.join(DATASET_DIR, "val/images")
VAL_LABELS_DIR = os.path.join(DATASET_DIR, "val/labels")
VISUALIZATION_DIR = os.path.join(DATASET_DIR, "visualization")

def convert_yolo_to_bbox(image_size, yolo_bbox):
    """将YOLO格式转换回普通边界框格式"""
    height, width = image_size
    x_center, y_center, w, h = yolo_bbox
    
    x1 = int((x_center - w/2) * width)
    y1 = int((y_center - h/2) * height)
    x2 = int((x_center + w/2) * width)
    y2 = int((y_center + h/2) * height)
    
    return [x1, y1, x2, y2]

def draw_bbox(image, bbox, color, label=None):
    """在图片上绘制边界框和标签"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    if label:
        # 添加填充的标签背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(image, (x1, y1-text_height-5), (x1+text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1-5), font, font_scale, (255,255,255), thickness)

def visualize_dataset(num_samples=10, include_val=True):
    """可视化数据集中的前n张图片"""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # 处理训练集
    train_images = sorted(os.listdir(TRAIN_IMAGES_DIR))[:num_samples]
    print(f"\n处理训练集图片...")
    for img_name in tqdm(train_images):
        image_path = os.path.join(TRAIN_IMAGES_DIR, img_name)
        label_path = os.path.join(TRAIN_LABELS_DIR, img_name.replace('.jpg', '.txt'))
        
        # 读取并处理图片
        process_image(image_path, label_path, "train")
    
    # 处理验证集
    if include_val:
        val_images = sorted(os.listdir(VAL_IMAGES_DIR))[:num_samples]
        print(f"\n处理验证集图片...")
        for img_name in tqdm(val_images):
            image_path = os.path.join(VAL_IMAGES_DIR, img_name)
            label_path = os.path.join(VAL_LABELS_DIR, img_name.replace('.jpg', '.txt'))
            
            # 读取并处理图片
            process_image(image_path, label_path, "val")

def process_image(image_path, label_path, prefix):
    """处理单张图片"""
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
        
    image_size = image.shape[:2]
    
    # 读取标注
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 绘制边界框
        for line in lines:
            class_id, *bbox = map(float, line.strip().split())
            bbox = convert_yolo_to_bbox(image_size, bbox)
            
            # 人物框用绿色，人脸框用蓝色
            if class_id == 0:  # person
                draw_bbox(image, bbox, (0,255,0), "person")
            else:  # face
                draw_bbox(image, bbox, (255,0,0), "face")
    
    # 保存可视化结果
    output_name = f"{prefix}_{os.path.basename(image_path)}"
    output_path = os.path.join(VISUALIZATION_DIR, output_name)
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # 可视化训练集和验证集的前20张图片
    visualize_dataset(20, include_val=True)
    print(f"\n可视化结果已保存到: {VISUALIZATION_DIR}") 