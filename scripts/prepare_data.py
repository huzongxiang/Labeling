import os
import json
import shutil
import zipfile
import subprocess
import xml.etree.ElementTree as ET
from tqdm import tqdm

# 常量定义
DATA_DIR = "datasets"
COCO_DIR = os.path.join(DATA_DIR, "COCO")
VOC_DIR = os.path.join(DATA_DIR, "VOC")
COMBINED_DIR = os.path.join(DATA_DIR, "combined")

COCO_URL = "http://images.cocodataset.org/zips/train2017.zip"
VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# 创建目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(COCO_DIR, exist_ok=True)
os.makedirs(VOC_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

def download_with_aria2(url, output_dir):
    """使用aria2下载文件"""
    filename = url.split("/")[-1]
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename} with aria2...")
        subprocess.run(["aria2c", "-x", "16", "-s", "16", "-d", output_dir, "-o", filename, url], check=True)

    return filepath

def extract_archive(filepath, output_dir, zip_format="zip"):
    """解压文件"""
    # 获取文件名（不含扩展名）作为解压目标文件夹
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    extract_path = os.path.join(output_dir, base_name)
    
    # 如果目标文件夹已存在且不为空，则跳过解压
    if os.path.exists(extract_path) and os.listdir(extract_path):
        print(f"目标文件夹已存在且不为空，跳过解压: {extract_path}")
        return extract_path
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在解压 {os.path.basename(filepath)}...")
    if zip_format == "zip":
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif zip_format == "tar":
        shutil.unpack_archive(filepath, output_dir)
    
    print(f"解压完成: {extract_path}")
    return extract_path

def extract_coco_person_annotations():
    """提取COCO数据集中的人物标注"""
    coco_annotations_path = os.path.join(COCO_DIR, "annotations", "instances_train2017.json")
    with open(coco_annotations_path) as f:
        coco_data = json.load(f)

    person_annotations = {}
    for annotation in tqdm(coco_data["annotations"], desc="Extracting COCO annotations"):
        if annotation["category_id"] == 1:  # 'person' category in COCO
            image_id = str(annotation["image_id"]).zfill(12)  # COCO格式需要补零
            if image_id not in person_annotations:
                person_annotations[image_id] = []
            # COCO格式是[x,y,width,height]，需要转换为[x1,y1,x2,y2]
            bbox = annotation["bbox"]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = x1 + bbox[2], y1 + bbox[3]
            person_annotations[image_id].append([x1, y1, x2, y2])

    return person_annotations, os.path.join(COCO_DIR, "train2017")

def extract_voc_person_annotations(voc_dir):
    """提取VOC数据集中的人物标注"""
    person_annotations = {}
    image_dir = os.path.join(voc_dir, "VOCdevkit", "VOC2007", "JPEGImages")
    annotation_dir = os.path.join(voc_dir, "VOCdevkit", "VOC2007", "Annotations")

    for annotation_file in tqdm(os.listdir(annotation_dir), desc="Extracting VOC annotations"):
        tree = ET.parse(os.path.join(annotation_dir, annotation_file))
        root = tree.getroot()
        image_id = root.find("filename").text.split('.')[0]

        bboxes = []
        for obj in root.findall("object"):
            if obj.find("name").text == "person":
                bbox = obj.find("bndbox")
                bboxes.append([
                    int(bbox.find("xmin").text),
                    int(bbox.find("ymin").text),
                    int(bbox.find("xmax").text),
                    int(bbox.find("ymax").text)
                ])

        if bboxes:
            person_annotations[image_id] = bboxes

    return person_annotations, image_dir

def main():
    """准备数据集"""
    if not os.path.exists(os.path.join(COMBINED_DIR, "combined_annotations.json")):
        # 下载数据集
        coco_filepath = download_with_aria2(COCO_URL, COCO_DIR)
        coco_annotations_filepath = download_with_aria2(COCO_ANNOTATIONS_URL, COCO_DIR)
        voc_filepath = download_with_aria2(VOC_URL, VOC_DIR)

        # 解压数据集
        extract_archive(coco_filepath, COCO_DIR, "zip")
        extract_archive(coco_annotations_filepath, COCO_DIR, "zip")  # 解压标注文件
        extract_archive(voc_filepath, VOC_DIR, "tar")

        # 提取标注
        person_annotations_coco, coco_image_dir = extract_coco_person_annotations()
        person_annotations_voc, voc_image_dir = extract_voc_person_annotations(VOC_DIR)

        # 合并数据集
        combined_annotations = {}
        combined_annotations.update(person_annotations_coco)
        combined_annotations.update(person_annotations_voc)
        combined_image_dirs = {**{k: coco_image_dir for k in person_annotations_coco},
                           **{k: voc_image_dir for k in person_annotations_voc}}
        
        # 保存合并数据集
        with open(os.path.join(COMBINED_DIR, "combined_annotations.json"), "w") as f:
            json.dump({
                "annotations": combined_annotations, 
                "image_dirs": combined_image_dirs
            }, f)
        print("数据集准备完成!")
    else:
        print("数据集已存在!")

if __name__ == "__main__":
    main() 