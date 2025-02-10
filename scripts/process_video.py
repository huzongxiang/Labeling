import os
import cv2
import yaml
import random
import shutil
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# 常量定义
VIDEO_PATH = "/home/hzx/Works/Datas/running/video_100_800.MP4"  # 替换为你的视频路径
OUTPUT_DIR = "./v2/datas"
TRAIN_IMAGES_DIR = os.path.join(OUTPUT_DIR, "train/images")
TRAIN_LABELS_DIR = os.path.join(OUTPUT_DIR, "train/labels")
VAL_IMAGES_DIR = os.path.join(OUTPUT_DIR, "val/images")
VAL_LABELS_DIR = os.path.join(OUTPUT_DIR, "val/labels")
SAMPLE_RATE = 2  # 每秒抽取一帧
VAL_RATIO = 0.1   # 验证集比例

model_face_path = "/home/hzx/Projects/COCO_VOC/yolov8l-face.pt"

def ensure_dirs():
    """确保所需目录存在"""
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
    os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(VAL_LABELS_DIR, exist_ok=True)

def extract_frames(video_path):
    """从视频中抽取帧"""
    print("开始抽帧...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"视频信息:")
    print(f"- 总帧数: {frame_count}")
    print(f"- FPS: {fps}")
    print(f"- 时长: {duration:.2f}秒")
    
    # 计算采样间隔
    sample_interval = fps // SAMPLE_RATE if fps > SAMPLE_RATE else 1
    estimated_frames = frame_count // sample_interval
    print(f"预计采样帧数: {estimated_frames}")
    
    extracted_frames = []
    frame_idx = 0
    
    with tqdm(total=estimated_frames, desc="抽帧进度") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_interval == 0:
                # 保存帧到临时目录
                frame_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append((frame_idx, frame_path))
                pbar.update(1)
            
            frame_idx += 1
    
    cap.release()
    print(f"抽帧完成，共抽取 {len(extracted_frames)} 帧")
    return extracted_frames

def split_dataset(frame_paths):
    """划分训练集和验证集"""
    random.shuffle(frame_paths)
    split_idx = int(len(frame_paths) * (1 - VAL_RATIO))
    
    train_frames = frame_paths[:split_idx]
    val_frames = frame_paths[split_idx:]
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_frames)} 帧")
    print(f"验证集: {len(val_frames)} 帧")
    
    return train_frames, val_frames

def convert_to_yolo_format(bbox, image_size):
    """将边界框转换为YOLO格式"""
    x1, y1, x2, y2 = bbox
    width, height = image_size
    
    x_center = (x1 + x2) / (2 * width)
    y_center = (y1 + y2) / (2 * height)
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    
    return [x_center, y_center, w, h]

def process_frames(frame_paths, is_training=True):
    """处理抽取的帧"""
    # 选择输出目录
    images_dir = TRAIN_IMAGES_DIR if is_training else VAL_IMAGES_DIR
    labels_dir = TRAIN_LABELS_DIR if is_training else VAL_LABELS_DIR
    
    # 初始化检测器
    person_detector = YOLO('yolov8l.pt')
    face_detector = YOLO(model_face_path)
    
    desc = "处理训练集" if is_training else "处理验证集"
    for frame_idx, frame_path in tqdm(frame_paths, desc=desc):
        # 读取图片
        image = cv2.imread(frame_path)
        if image is None:
            print(f"无法读取图片: {frame_path}")
            continue
        
        image_size = (image.shape[1], image.shape[0])
        
        # 检测人物
        person_results = person_detector(image, imgsz=640)[0]
        person_boxes = []
        for r in person_results.boxes:
            if r.cls == 0 and r.conf > 0.5:
                person_boxes.append(r.xyxy[0].cpu().numpy())
        
        # 检测人脸
        face_results = face_detector(image, imgsz=640)[0]
        face_boxes = []
        if len(face_results.boxes):
            confident_masks = face_results.boxes.conf > 0.5
            if confident_masks.any():
                face_boxes = face_results.boxes.xyxy[confident_masks].cpu().numpy()
        
        # 只处理有检测结果的帧
        if len(person_boxes) > 0 or len(face_boxes) > 0:
            # 移动图片到对应目录
            new_image_path = os.path.join(images_dir, f"frame_{frame_idx:06d}.jpg")
            shutil.move(frame_path, new_image_path)
            
            # 生成标签文件
            label_path = os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt")
            with open(label_path, 'w') as f:
                for bbox in person_boxes:
                    yolo_bbox = convert_to_yolo_format(bbox, image_size)
                    f.write(f"0 {' '.join(map(str, yolo_bbox))}\n")
                
                for bbox in face_boxes:
                    yolo_bbox = convert_to_yolo_format(bbox, image_size)
                    f.write(f"1 {' '.join(map(str, yolo_bbox))}\n")
        else:
            # 删除没有检测结果的帧
            os.remove(frame_path)

def create_dataset_yaml():
    """创建数据集的yaml配置文件"""
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'person',
            1: 'face'
        }
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

def main():
    """主函数"""
    # 确保目录存在
    ensure_dirs()
    
    # 抽取视频帧
    extracted_frames = extract_frames(VIDEO_PATH)
    
    # 划分数据集
    train_frames, val_frames = split_dataset(extracted_frames)
    
    # 处理训练集和验证集
    process_frames(train_frames, is_training=True)
    process_frames(val_frames, is_training=False)
    
    # 创建数据集配置文件
    create_dataset_yaml()
    
    print("\n处理完成！")
    print(f"训练集图片: {TRAIN_IMAGES_DIR}")
    print(f"训练集标签: {TRAIN_LABELS_DIR}")
    print(f"验证集图片: {VAL_IMAGES_DIR}")
    print(f"验证集标签: {VAL_LABELS_DIR}")
    print(f"数据集配置文件: {os.path.join(OUTPUT_DIR, 'dataset.yaml')}")

if __name__ == "__main__":
    main() 