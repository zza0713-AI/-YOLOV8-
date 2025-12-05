import cv2
import numpy as np
import torch
import os

# 图像预处理函数
def preprocess_image(image_path, target_size=640, augment=False, save=True):
    print(f"Image path: {image_path}")  # 打印路径确认
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 数据增强（可选）
    if augment:
        # 随机水平翻转
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
        # 可以添加更多增强方法，如旋转、裁剪、亮度调整等
        
    # 调整图像大小（保持宽高比，填充至目标尺寸）
    image = letterbox(image, target_size)

    # 保存处理后的图像到同目录下的 pre_photo 文件夹（可选）
    if save:
        try:
            dir_path = os.path.dirname(image_path)
            pre_dir = os.path.join(dir_path, 'pre_photo')
            os.makedirs(pre_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            save_path = os.path.join(pre_dir, base_name)
            # letterbox 返回的是 RGB uint8 图像，确保格式正确后保存为 BGR
            save_img = image.copy()
            save_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_bgr)
            print(f"Saved preprocessed image to {save_path}")
        except Exception as e:
            print(f"Failed to save preprocessed image: {e}")
    
    # 归一化到[0, 1]
    image = image / 255.0
    
    # 转换为PyTorch张量 (CHW)
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # 转换为CHW格式
    
    # 增加一个batch维度
    image_tensor = image_tensor.unsqueeze(0)  # shape: (1, C, H, W)
    
    return image_tensor

# 填充图像至目标大小（保持宽高比）
def letterbox(image, target_size=640):
    h, w, _ = image.shape
    scale = target_size / max(h, w)  # 缩放因子
    nh, nw = int(h * scale), int(w * scale)
    
    # 调整图像大小
    resized_image = cv2.resize(image, (nw, nh))
    
    # 创建目标大小的空白图像（黑色填充）
    padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 填充图像
    padded_image[(target_size - nh) // 2:(target_size - nh) // 2 + nh,
                 (target_size - nw) // 2:(target_size - nw) // 2 + nw] = resized_image
    
    return padded_image

# 示例：使用函数加载并预处理图像
image_path =r"D:\\study_opencv\\new_project\\bus_photo\\e30e5d21d2be626f3dc7ab41e9a58c10.jpg"
processed_image = preprocess_image(image_path, target_size=640, augment=True)
# 打印输出图像张量形状
if processed_image is not None:
    print(f"Processed image shape: {processed_image.shape}")
