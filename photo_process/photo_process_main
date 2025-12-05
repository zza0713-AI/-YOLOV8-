import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
from collections import defaultdict

# 创建输出目录
output_dir = "output_photo"
os.makedirs(output_dir, exist_ok=True)

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')

# 定义类别ID
person_class_id = 0
bicycle_class_id = 1
car_class_id = 2
bus_class_id = 5

# 使用英文类别名称
class_names = {
    person_class_id: "Person",
    bicycle_class_id: "Bicycle", 
    car_class_id: "Car",
    bus_class_id: "Bus"
}

# 颜色映射
color_map = {
    person_class_id: (0, 255, 0),      # 绿色 - 行人
    bicycle_class_id: (255, 0, 0),     # 蓝色 - 自行车
    car_class_id: (0, 0, 255),         # 红色 - 小轿车
    bus_class_id: (0, 165, 255)        # 橙色 - 公共汽车
}

def detect_objects(image_path, model, conf_threshold=0.25):
    """
    使用YOLOv8模型检测图像中的对象
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    results = model(image, conf=conf_threshold, verbose=False)
    counts = defaultdict(int)
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                if class_id in [person_class_id, bicycle_class_id, car_class_id, bus_class_id]:
                    counts[class_id] += 1
                    
                    color = color_map.get(class_id, (255, 255, 255))
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # 使用英文标签
                    label = f"{class_names[class_id]} {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                                 (int(x1) + label_size[0], int(y1)), color, -1)
                    cv2.putText(image, label, (int(x1), int(y1) - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image, counts

def add_statistics_to_image(image, counts):
    """
    在图像上添加统计信息（使用英文）
    """
    total_persons = counts.get(person_class_id, 0)
    total_vehicles = (counts.get(bicycle_class_id, 0) + 
                     counts.get(car_class_id, 0) + 
                     counts.get(bus_class_id, 0))
    
    # 创建统计信息背景
    stats_background = np.zeros((100, image.shape[1], 3), dtype=np.uint8)
    
    # 使用英文统计信息
    texts = [
        "Detection Statistics:",
        f"Persons: {total_persons}",
        f"Bicycles: {counts.get(bicycle_class_id, 0)}",
        f"Cars: {counts.get(car_class_id, 0)}", 
        f"Buses: {counts.get(bus_class_id, 0)}",
        f"Total Vehicles: {total_vehicles}",
        f"Total Objects: {total_persons + total_vehicles}"
    ]
    
    for i, text in enumerate(texts):
        y_position = 20 + i * 15
        cv2.putText(stats_background, text, (10, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    result_image = np.vstack([image, stats_background])
    return result_image

def process_images_from_folder(folder_path, model):
    """
    处理文件夹中的所有图像
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for filename in os.listdir(folder_path):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in supported_formats:
            image_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            
            annotated_image, counts = detect_objects(image_path, model)
            
            if annotated_image is not None and counts is not None:
                result_image = add_statistics_to_image(annotated_image, counts)
                output_path = os.path.join(output_dir, f"detected_{filename}")
                cv2.imwrite(output_path, result_image)
                print(f"Saved: {output_path}")
                
                # 打印统计信息（使用中文）
                print(f"检测统计 - 行人: {counts.get(person_class_id, 0)}, "
                      f"自行车: {counts.get(bicycle_class_id, 0)}, "
                      f"小轿车: {counts.get(car_class_id, 0)}, "
                      f"公共汽车: {counts.get(bus_class_id, 0)}")
            else:
                print(f"Failed to process: {filename}")
            
            print("-" * 50)

def main():
    preprocessed_folder = r"D:\\study_opencv\\new_project\\bus_photo\\pre_photo"
    
    if not os.path.exists(preprocessed_folder):
        print(f"预处理图像文件夹不存在: {preprocessed_folder}")
        return
    
    print("开始目标检测...")
    process_images_from_folder(preprocessed_folder, model)
    print("所有图像处理完成！")

if __name__ == "__main__":
    main()
