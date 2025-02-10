import os
import json
import shutil
from pathlib import Path
import random

def prepare_yolo_dataset(
    coco_dir="/home/hzx/Projects/COCO_VOC/datasets/COCO/train2017",
    voc_dir="/home/hzx/Projects/COCO_VOC/datasets/VOC/VOCdevkit/VOC2007/JPEGImages",
    labels_dir="/home/hzx/Projects/COCO_VOC/datasets/YOLO_labels",
    output_dir="datasets",
    train_ratio=0.8
):
    # 创建YOLO所需的目录结构
    dataset_dir = Path(output_dir)/"train_data"
    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 获取所有标签文件
    label_files = list(Path(labels_dir).glob('*.txt'))
    image_ids = [f.stem for f in label_files]
    
    # 随机划分训练集和验证集
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * train_ratio)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]

    def get_image_path(image_id):
        """根据图片ID判断图片位置"""
        # 先检查COCO目录
        coco_path = os.path.join(coco_dir, f"{image_id}.jpg")
        if os.path.exists(coco_path):
            return coco_path
            
        # 再检查VOC目录
        voc_path = os.path.join(voc_dir, f"{image_id}.jpg")
        if os.path.exists(voc_path):
            return voc_path
            
        return None

    # 复制文件到对应目录
    def copy_files(ids, split):
        for image_id in ids:
            # 获取源图片路径
            src_img = get_image_path(image_id)
            if src_img is None:
                print(f"警告: 找不到图片 {image_id}")
                continue
                
            # 复制图片
            dst_img = dataset_dir / 'images' / split / f"{image_id}.jpg"
            
            # 复制标签
            src_label = os.path.join(labels_dir, f"{image_id}.txt")
            dst_label = dataset_dir / 'labels' / split / f"{image_id}.txt"
            
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_label, dst_label)

    # 复制训练集和验证集文件
    print("正在复制训练集文件...")
    copy_files(train_ids, 'train')
    print("正在复制验证集文件...")
    copy_files(val_ids, 'val')

    # 创建data.yaml文件
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: person
  1: face

nc: 2
"""
    
    with open(dataset_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content.strip())

    # 统计实际复制的文件数量
    train_count = len(list((dataset_dir / 'images' / 'train').glob('*.jpg')))
    val_count = len(list((dataset_dir / 'images' / 'val').glob('*.jpg')))

    print(f"\n数据集准备完成:")
    print(f"训练集数量: {train_count}")
    print(f"验证集数量: {val_count}")
    print(f"数据配置文件保存在: {dataset_dir}/data.yaml")

if __name__ == "__main__":
    prepare_yolo_dataset()