import os
import cv2
import yaml
import random
import shutil
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

# 从配置文件加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 常量定义
VIDEO_PATH = config['video_path']
OUTPUT_DIR = config['output_dir']
TRAIN_IMAGES_DIR = os.path.join(OUTPUT_DIR, "train/images")
TRAIN_LABELS_DIR = os.path.join(OUTPUT_DIR, "train/labels")
VAL_IMAGES_DIR = os.path.join(OUTPUT_DIR, "val/images")
VAL_LABELS_DIR = os.path.join(OUTPUT_DIR, "val/labels")
SAMPLE_RATE = config['sample_rate']
VAL_RATIO = config['val_ratio']

# 模型路径和配置
MODEL_PATHS = config['model_paths']
MODEL_CONFIGS = config.get('model_configs', {})

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

def init_models(config):
    """初始化所有模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    
    # 初始化 YOLO 模型
    if "yolo_person" in config['model_paths']:
        models['yolo_person'] = YOLO(config['model_paths']['yolo_person'])
    if "yolo_face" in config['model_paths']:
        models['yolo_face'] = YOLO(config['model_paths']['yolo_face'])
    
    # 初始化 SAM 模型
    if "sam" in config['model_paths']:
        sam_config = config['model_configs']['sam']
        sam_model = build_sam2(sam_config['model_cfg'], 
                             config['model_paths']['sam'], 
                             device=sam_config['device'])
        models['sam'] = SAM2ImagePredictor(sam_model)
    
    # 初始化 GroundingDINO 模型
    if "groundingdino" in config['model_paths']:
        dino_config = config['model_configs']['groundingdino']
        processor = AutoProcessor.from_pretrained(config['model_paths']['groundingdino'])
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            config['model_paths']['groundingdino']
        ).to(device)
        models['groundingdino'] = (processor, model)
    
    return models, device

def detect_objects(image, model_name, models, device):
    """使用指定模型检测对象"""
    if model_name.startswith("yolo"):
        model = models[model_name]
        results = model(image, imgsz=config['detection']['image_size'])[0]
        boxes = []
        for r in results.boxes:
            if r.conf > config['detection']['confidence_threshold']:
                boxes.append((r.cls, r.xyxy[0].cpu().numpy()))
        return boxes
    
    elif model_name == "sam":
        model = models[model_name]
        model.set_image(np.array(image))
        
        # 这里需要输入提示（点、框或mask）
        masks, scores, logits = model.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=False,
        )
        
        # 处理masks
        if masks.ndim == 2:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
            
        return masks, scores, logits
    
    elif model_name == "groundingdino":
        processor, model = models[model_name]
        dino_config = config['model_configs']['groundingdino']
        
        # 处理输入
        inputs = processor(
            images=Image.fromarray(image) if isinstance(image, np.ndarray) else image,
            text=dino_config['text_prompts'],
            return_tensors="pt"
        ).to(device)
        
        # 进行检测
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 后处理结果
        results = processor.post_process_object_detection(
            outputs,
            threshold=dino_config['box_threshold'],
            target_sizes=[(image.shape[0], image.shape[1])]
        )
        
        return results[0]
    
    return []

def process_frames(frame_paths, is_training=True):
    """处理抽取的帧"""
    # 初始化模型
    models, device = init_models(config)
    
    # 选择输出目录
    images_dir = TRAIN_IMAGES_DIR if is_training else VAL_IMAGES_DIR
    labels_dir = TRAIN_LABELS_DIR if is_training else VAL_LABELS_DIR
    
    desc = "处理训练集" if is_training else "处理验证集"
    for frame_idx, frame_path in tqdm(frame_paths, desc=desc):
        # 读取图片
        image = cv2.imread(frame_path)
        if image is None:
            print(f"无法读取图片: {frame_path}")
            continue
        
        image_size = (image.shape[1], image.shape[0])
        all_boxes = []
        
        # 使用每个模型进行检测
        for model_name in MODEL_PATHS.keys():
            results = detect_objects(image, model_name, models, device)
            
            if model_name.startswith("yolo"):
                all_boxes.extend(results)
            elif model_name == "groundingdino":
                # 处理GroundingDINO的结果
                for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                    if score > config['detection']['confidence_threshold']:
                        all_boxes.append((label, box.cpu().numpy()))
            elif model_name == "sam":
                # 处理SAM的结果
                masks, scores, _ = results
                # 这里可以根据需要处理SAM的mask结果
                pass
        
        # 处理检测结果
        if all_boxes:
            # 移动图片到对应目录
            new_image_path = os.path.join(images_dir, f"frame_{frame_idx:06d}.jpg")
            shutil.move(frame_path, new_image_path)
            
            # 生成标签文件
            label_path = os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt")
            with open(label_path, 'w') as f:
                for class_id, bbox in all_boxes:
                    yolo_bbox = convert_to_yolo_format(bbox, image_size)
                    f.write(f"{int(class_id)} {' '.join(map(str, yolo_bbox))}\n")
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
            1: 'face',
            # 添加更多类别名称
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
    print(f"数据集配置文件: {os.path.join(OUTPUT_DIR, 'data.yaml')}")

if __name__ == "__main__":
    main() 