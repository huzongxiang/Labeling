import os
import cv2
import yaml
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

DATA_DIR = "datasets"
COMBINED_DIR = os.path.join(DATA_DIR, "combined")
YOLO_LABELS_DIR = os.path.join(DATA_DIR, "YOLO_labels")

class PersonDataset(Dataset):
    def __init__(self, annotations, image_dirs):
        self.image_ids = list(annotations.keys())
        self.annotations = annotations
        self.image_dirs = image_dirs

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_dir = self.image_dirs[image_id]

        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        person_bboxes = self.annotations[image_id]
        
        # 返回原始图像尺寸，用于后续坐标转换
        return image_id, image, person_bboxes, (image.shape[0], image.shape[1])

def convert_to_yolo_format(image_size, bbox):
    """将边界框转换为YOLO格式"""
    x_center = (bbox[0] + bbox[2]) / 2.0 / image_size[1]
    y_center = (bbox[1] + bbox[3]) / 2.0 / image_size[0]
    width = (bbox[2] - bbox[0]) / image_size[1]
    height = (bbox[3] - bbox[1]) / image_size[0]
    return x_center, y_center, width, height

def process_batch(dataloader, output_labels_dir, label_map, face_detector):
    """批量处理并生成标签"""
    os.makedirs(output_labels_dir, exist_ok=True)
    
    for batch in tqdm(dataloader, desc="处理批次"):
        image_ids, images, person_bboxes_batch, image_sizes = batch
        
        # 直接对整批次图片进行人脸检测
        results = face_detector(images, imgsz=640)
        
        # 处理每张图片的检测结果
        for image_id, person_boxes, result, img_size in zip(image_ids, person_bboxes_batch, results, image_sizes):
            # 获取人脸检测结果
            face_boxes = []
            if len(result.boxes):
                # 只保留置信度大于0.5的检测结果
                confident_masks = result.boxes.conf > 0.5
                if confident_masks.any():
                    detected_faces = result.boxes.xyxy[confident_masks].cpu().numpy()
                    
                    # 筛选在人物框内的人脸
                    for face_box in detected_faces:
                        face_center_x = (face_box[0] + face_box[2]) / 2
                        face_center_y = (face_box[1] + face_box[3]) / 2
                        
                        # 检查人脸中心点是否在任一人物框内
                        for person_box in person_boxes:
                            if (face_center_x >= person_box[0] and face_center_x <= person_box[2] and
                                face_center_y >= person_box[1] and face_center_y <= person_box[3]):
                                face_boxes.append(face_box)
                                break
            
            # 写入YOLO格式标签
            label_file_path = os.path.join(output_labels_dir, f"{image_id}.txt")
            with open(label_file_path, "w") as f:
                # 写入人物框
                for bbox in person_boxes:
                    yolo_bbox = convert_to_yolo_format(img_size, bbox)
                    f.write(f"{label_map['person']} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")
                
                # 写入人脸框（已经过筛选，确保在人物框内）
                for face_bbox in face_boxes:
                    yolo_bbox = convert_to_yolo_format(img_size, face_bbox)
                    f.write(f"{label_map['face']} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")

def collate_fn(batch):
    """自定义批处理函数，处理变长的边界框列表"""
    image_ids, images, person_bboxes, image_sizes = zip(*batch)
    return list(image_ids), list(images), list(person_bboxes), list(image_sizes)

def main():
    """主函数"""
    # 加载合并数据
    with open(os.path.join(COMBINED_DIR, "combined_annotations_voc.json"), "r") as f:
        combined_data = json.load(f)
        combined_annotations = combined_data["annotations"]
        combined_image_dirs = combined_data["image_dirs"]

    # 打印数据集统计信息
    total_images = len(combined_annotations)
    total_person_boxes = sum(len(boxes) for boxes in combined_annotations.values())
    print(f"\n数据集统计信息:")
    print(f"总图片数量: {total_images}")
    print(f"总人物框数量: {total_person_boxes}")
    print(f"平均每张图片人物框数量: {total_person_boxes / total_images:.2f}\n")

    # 初始化YOLO人脸检测器
    face_detector = YOLO('yolov8l-face.pt')
    
    # 创建数据加载器
    dataset = PersonDataset(combined_annotations, combined_image_dirs)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 处理数据集
    LABEL_MAP = {"person": 0, "face": 1}
    process_batch(dataloader, YOLO_LABELS_DIR, LABEL_MAP, face_detector)

if __name__ == "__main__":
    main() 