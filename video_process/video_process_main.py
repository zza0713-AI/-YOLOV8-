"""
YOLOv8 + ByteTrack 行人和车辆检测与追踪程序 (带速度计算、摄像头运动补偿和拥堵检测)
改进：
1. 使用更大的模型 (yolov8l) 改善远距离检测
2. 使用ByteTrack追踪，避免重复计数
3. 每个对象只计数一次（通过唯一的track_id）
4. 计算行人和车辆的实时速度
5. 检测摄像头运动，自动补偿
6. ✅ 检测速度急剧下降，判断道路拥堵等级
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict
import math
from enum import Enum
from PIL import Image, ImageDraw, ImageFont

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CongestionLevel(Enum):
    """道路拥堵等级枚举"""
    SMOOTH = "smooth"          # 畅通 (绿色)
    LIGHT = "light"            # 轻微拥堵 (黄色)
    MODERATE = "moderate"      # 中等拥堵 (橙色)
    HEAVY = "heavy"            # 严重拥堵 (红色)
    SEVERE = "severe"          # 极度拥堵 (深红色)


class CongestionDetector:
    """道路拥堵检测器"""
    
    def __init__(self, 
                 window_size=30,
                 speed_drop_threshold=0.4,
                 density_threshold=0.15):
        """
        初始化拥堵检测器
        
        Args:
            window_size: 时间窗口大小（帧数）
            speed_drop_threshold: 速度下降阈值比例 (0-1)
            density_threshold: 密度阈值 (0-1)
        """
        self.window_size = window_size
        self.speed_drop_threshold = speed_drop_threshold
        self.density_threshold = density_threshold
        
        # 统计信息
        self.speed_history = defaultdict(list)  # 每个对象的速度历史
        self.object_count_history = []  # 每帧的对象数量
        self.speed_drop_events = []  # 速度急剧下降事件
        self.frame_congestion_levels = []  # 每帧的拥堵等级
    
    def calculate_speed_drop_ratio(self, speeds):
        """
        计算速度下降比例
        
        Args:
            speeds: 速度列表 (最近的窗口内的速度)
        
        Returns:
            (下降比例, 平均速度)
        """
        if len(speeds) < 2:
            return 0, 0
        
        # 使用前半段和后半段的平均速度比较
        mid = len(speeds) // 2
        avg_first_half = np.mean(speeds[:mid]) if mid > 0 else speeds[0]
        avg_second_half = np.mean(speeds[mid:])
        
        if avg_first_half > 0:
            drop_ratio = (avg_first_half - avg_second_half) / avg_first_half
        else:
            drop_ratio = 0
        
        return max(0, drop_ratio), avg_second_half
    
    def calculate_traffic_density(self, current_object_count, frame_area):
        """
        计算交通密度
        
        Args:
            current_object_count: 当前帧的对象数量
            frame_area: 帧的像素面积
        
        Returns:
            密度值 (0-1)
        """
        if frame_area == 0:
            return 0
        
        # 密度 = 对象数量 / (帧面积 / 标准化系数)
        # 标准化系数使密度在合理范围内
        normalization_factor = frame_area / 100000  # 100000像素为基准
        density = min(1.0, (current_object_count / max(1, normalization_factor)))
        
        return density
    
    def detect_congestion_level(self, speeds, object_count, frame_area):
        """
        检测道路拥堵等级
        
        Args:
            speeds: 最近的速度数据
            object_count: 当前交通对象数量
            frame_area: 帧的像素面积
        
        Returns:
            (拥堵等级, 详细信息字典)
        """
        details = {
            'avg_speed': 0,
            'speed_drop_ratio': 0,
            'traffic_density': 0,
            'congestion_factors': []
        }
        
        # 计算平均速度和速度下降
        if speeds:
            details['avg_speed'] = np.mean(speeds)
            drop_ratio, _ = self.calculate_speed_drop_ratio(speeds)
            details['speed_drop_ratio'] = drop_ratio
        
        # 计算交通密度
        density = self.calculate_traffic_density(object_count, frame_area)
        details['traffic_density'] = density
        
        # 判断拥堵等级的因素
        congestion_score = 0
        
        # 1. 速度因素 (权重: 40%)
        if details['avg_speed'] < 1.0:  # 低于3.6 km/h
            congestion_score += 0.4
            details['congestion_factors'].append('极低速度')
        elif details['avg_speed'] < 2.5:  # 低于9 km/h
            congestion_score += 0.3
            details['congestion_factors'].append('低速度')
        elif details['avg_speed'] < 5.0:  # 低于18 km/h
            congestion_score += 0.15
            details['congestion_factors'].append('中等速度')
        
        # 2. 速度下降因素 (权重: 30%)
        if details['speed_drop_ratio'] > 0.5:  # 速度下降超过50%
            congestion_score += 0.3
            details['congestion_factors'].append('速度急剧下降')
        elif details['speed_drop_ratio'] > 0.3:  # 速度下降超过30%
            congestion_score += 0.2
            details['congestion_factors'].append('速度显著下降')
        elif details['speed_drop_ratio'] > 0.1:  # 速度下降超过10%
            congestion_score += 0.1
            details['congestion_factors'].append('速度略有下降')
        
        # 3. 密度因素 (权重: 30%)
        if density > 0.8:  # 密度非常高
            congestion_score += 0.3
            details['congestion_factors'].append('交通密度极高')
        elif density > 0.5:  # 密度很高
            congestion_score += 0.2
            details['congestion_factors'].append('交通密度很高')
        elif density > self.density_threshold:  # 密度较高
            congestion_score += 0.1
            details['congestion_factors'].append('交通密度较高')
        
        # 根据综合得分判断拥堵等级
        if congestion_score < 0.15:
            level = CongestionLevel.SMOOTH
        elif congestion_score < 0.35:
            level = CongestionLevel.LIGHT
        elif congestion_score < 0.55:
            level = CongestionLevel.MODERATE
        elif congestion_score < 0.75:
            level = CongestionLevel.HEAVY
        else:
            level = CongestionLevel.SEVERE
        
        details['congestion_score'] = congestion_score
        details['level'] = level.value
        
        return level, details


class CameraMotionDetector:
    """摄像头运动检测器 - 使用特征点追踪"""
    
    def __init__(self, max_corners=200, quality_level=0.01, min_distance=10):
        """
        初始化摄像头运动检测器
        
        Args:
            max_corners: 最多检测的角点数
            quality_level: 角点质量阈值
            min_distance: 角点之间的最小距离
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.prev_gray = None
        self.prev_points = None
    
    def detect_motion(self, frame):
        """
        检测摄像头运动
        
        Returns:
            (camera_dx, camera_dy) - 摄像头在x和y方向的像素移动量
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        camera_dx, camera_dy = 0, 0
        
        # 检测角点
        corners = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7,
            useHarrisDetector=False
        )
        
        if corners is not None and self.prev_points is not None:
            try:
                # 使用Lucas-Kanade光流算法追踪特征点
                next_points, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_points, None,
                    winSize=(15, 15),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                # 筛选出好的追踪点
                if status is not None:
                    good_old = self.prev_points[status == 1]
                    good_new = next_points[status == 1]
                    
                    if len(good_old) > 10:  # 至少要有10个可靠的追踪点
                        # 计算所有点的平均运动
                        movements = good_new - good_old
                        camera_dx = np.median(movements[:, 0])
                        camera_dy = np.median(movements[:, 1])
            except Exception as e:
                logger.debug(f"光流计算出错: {e}")
                camera_dx, camera_dy = 0, 0
        
        # 保存当前帧的信息用于下一帧
        self.prev_gray = gray
        self.prev_points = corners
        
        return camera_dx, camera_dy


class YOLOTrackerDetector:
    """YOLOv8 + ByteTrack 行人和车辆检测追踪器（带拥堵检测）"""
    
    # COCO数据集中的类别ID映射
    CLASS_NAMES = {
        0: 'person',      # 行人
        1: 'bicycle',     # 自行车
        2: 'car',         # 小轿车
        3: 'motorcycle',  # 摩托车
        5: 'bus',         # 公共汽车
        7: 'truck',       # 卡车
    }
    
    # 类别到我们需要的类别的映射
    CLASS_MAPPING = {
        'person': 'pedestrian',        # 行人
        'bicycle': 'bicycle',          # 自行车
        'car': 'car',                  # 小轿车
        'motorcycle': 'bicycle',       # 摩托车归为自行车
        'bus': 'bus',                  # 公共汽车
        'truck': 'car',                # 卡车归为小轿车
    }
    
    # 颜色配置 (BGR)
    COLORS = {
        'pedestrian': (0, 255, 0),     # 绿色
        'bicycle': (255, 0, 0),        # 蓝色
        'car': (0, 0, 255),            # 红色
        'bus': (0, 165, 255),          # 橙色
    }
    
    # 拥堵等级颜色配置
    CONGESTION_COLORS = {
        CongestionLevel.SMOOTH: (0, 255, 0),      # 绿色
        CongestionLevel.LIGHT: (0, 255, 255),     # 黄色
        CongestionLevel.MODERATE: (0, 165, 255),  # 橙色
        CongestionLevel.HEAVY: (0, 0, 255),       # 红色
        CongestionLevel.SEVERE: (0, 0, 139),      # 深红色
    }
    
    # 拥堵等级中文名称
    CONGESTION_NAMES_CN = {
        CongestionLevel.SMOOTH: '畅通',
        CongestionLevel.LIGHT: '轻微拥堵',
        CongestionLevel.MODERATE: '中等拥堵',
        CongestionLevel.HEAVY: '严重拥堵',
        CongestionLevel.SEVERE: '极度拥堵',
    }
    
    # 拥堵等级英文名称（备用）
    CONGESTION_NAMES_EN = {
        CongestionLevel.SMOOTH: 'Smooth',
        CongestionLevel.LIGHT: 'Light Congestion',
        CongestionLevel.MODERATE: 'Moderate Congestion',
        CongestionLevel.HEAVY: 'Heavy Congestion',
        CongestionLevel.SEVERE: 'Severe Congestion',
    }
    
    def __init__(self, 
                 model_name='yolov8l.pt',
                 conf_threshold=0.3,
                 iou_threshold=0.5,
                 output_dir='output_video',
                 input_dir='pre_video',
                 tracker='bytetrack.yaml',
                 pixels_per_meter=50,
                 enable_camera_compensation=True,
                 enable_congestion_detection=True,
                 use_chinese_text=True):
        """
        初始化检测追踪器
        
        Args:
            model_name: YOLOv8模型名称
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU阈值
            output_dir: 输出目录
            input_dir: 输入视频目录
            tracker: 追踪算法
            pixels_per_meter: 像素与米的转换比例
            enable_camera_compensation: 是否启用摄像头运动补偿
            enable_congestion_detection: 是否启用拥堵检测
            use_chinese_text: 是否使用中文文本显示
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.tracker_type = tracker
        self.pixels_per_meter = pixels_per_meter
        self.enable_camera_compensation = enable_camera_compensation
        self.enable_congestion_detection = enable_congestion_detection
        self.use_chinese_text = use_chinese_text
        
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/stats").mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        logger.info(f"加载YOLOv8模型: {model_name}")
        self.model = YOLO(model_name)
        logger.info("模型加载完成")
        
        # 初始化摄像头运动检测器
        self.camera_motion_detector = CameraMotionDetector()
        
        # 初始化拥堵检测器
        self.congestion_detector = CongestionDetector(
            window_size=30,
            speed_drop_threshold=0.4,
            density_threshold=0.15
        )
        
        # 尝试加载中文字体
        self.font_path = None
        if self.use_chinese_text:
            # 常见的中文字体路径
            possible_fonts = [
                # Windows
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
                # Linux
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                # macOS
                "/Library/Fonts/Arial Unicode.ttf",
                "/System/Library/Fonts/PingFang.ttc",
            ]
            
            for font_path in possible_fonts:
                if os.path.exists(font_path):
                    self.font_path = font_path
                    logger.info(f"找到中文字体: {font_path}")
                    break
            
            if self.font_path is None:
                logger.warning("未找到中文字体，将使用英文显示")
                self.use_chinese_text = False
        
        # 摄像头运动统计
        self.camera_stats = {
            'motions': [],
            'total_motion': (0, 0),
        }
        
        # 追踪统计信息
        self.track_stats = {
            'total_frames': 0,
            'unique_pedestrians': set(),
            'unique_bicycles': set(),
            'unique_cars': set(),
            'unique_buses': set(),
            'track_history': defaultdict(list),
            'track_speeds_absolute': defaultdict(list),
            'track_speeds_relative': defaultdict(list),
            'frame_detections': [],
            'max_speeds_absolute': defaultdict(float),
            'max_speeds_relative': defaultdict(float),
            'avg_speeds_absolute': defaultdict(list),
            'avg_speeds_relative': defaultdict(list),
            'congestion_history': [],  # 每帧的拥堵信息
        }
    
    def get_class_label(self, class_id):
        """获取类别标签"""
        if class_id in self.CLASS_NAMES:
            original_name = self.CLASS_NAMES[class_id]
            mapped_name = self.CLASS_MAPPING.get(original_name, original_name)
            return mapped_name
        return None
    
    def calculate_distance(self, p1, p2):
        """计算两点之间的欧几里得距离（像素）"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def calculate_speed(self, distance_pixels, fps, pixels_per_meter=50):
        """计算速度"""
        time_interval = 1.0 / fps
        distance_meters = distance_pixels / pixels_per_meter
        speed_ms = distance_meters / time_interval
        speed_kmh = speed_ms * 3.6
        return speed_ms, speed_kmh
    
    def compensate_camera_motion(self, object_speed_ms, camera_motion, fps, pixels_per_meter):
        """补偿摄像头运动，计算对象的相对速度"""
        camera_motion_pixels = math.sqrt(camera_motion[0]**2 + camera_motion[1]**2)
        camera_speed_ms, _ = self.calculate_speed(camera_motion_pixels, fps, pixels_per_meter)
        relative_speed_ms = object_speed_ms - camera_speed_ms
        return relative_speed_ms, camera_speed_ms
    
    def put_chinese_text(self, image, text, position, font_size=20, color=(255, 255, 255)):
        """
        在图像上绘制中文文本
        
        Args:
            image: OpenCV图像 (BGR格式)
            text: 要绘制的中文文本
            position: 文本位置 (x, y)
            font_size: 字体大小
            color: 文本颜色 (BGR格式)
        
        Returns:
            绘制文本后的图像
        """
        if not self.use_chinese_text or self.font_path is None:
            # 使用英文替代
            english_text = self.CONGESTION_NAMES_EN.get(text, text)
            cv2.putText(image, english_text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_size/20, color, 2)
            return image
        
        try:
            # 将OpenCV图像转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # 加载中文字体
            font = ImageFont.truetype(self.font_path, font_size)
            
            # 绘制文本
            draw.text(position, text, font=font, fill=color)
            
            # 将PIL图像转换回OpenCV格式
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"绘制中文文本失败: {e}, 使用英文替代")
            english_text = self.CONGESTION_NAMES_EN.get(text, text)
            cv2.putText(image, english_text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_size/20, color, 2)
        
        return image
    
    def draw_detections_with_tracking(self, frame, results, fps, camera_motion):
        """在帧上绘制检测结果、追踪信息和速度"""
        frame_stats = {
            'pedestrian': [],
            'bicycle': [],
            'car': [],
            'bus': [],
            'detections': [],
            'camera_motion': camera_motion,
        }
        
        if results is None or len(results) == 0:
            return frame, frame_stats
        
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = None
                
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])
                
                if conf < self.conf_threshold:
                    continue
                
                class_label = self.get_class_label(class_id)
                if class_label is None:
                    continue
                
                if track_id is not None:
                    if class_label == 'pedestrian':
                        self.track_stats['unique_pedestrians'].add(track_id)
                    elif class_label == 'bicycle':
                        self.track_stats['unique_bicycles'].add(track_id)
                    elif class_label == 'car':
                        self.track_stats['unique_cars'].add(track_id)
                    elif class_label == 'bus':
                        self.track_stats['unique_buses'].add(track_id)
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                speed_ms_absolute = 0
                speed_kmh_absolute = 0
                speed_ms_relative = 0
                speed_kmh_relative = 0
                camera_speed_ms = 0
                
                if track_id is not None:
                    history = self.track_stats['track_history'][track_id]
                    
                    if len(history) > 0:
                        last_position = history[-1]
                        distance_pixels = self.calculate_distance(last_position, center)
                        speed_ms_absolute, speed_kmh_absolute = self.calculate_speed(
                            distance_pixels, fps, self.pixels_per_meter
                        )
                        
                        if self.enable_camera_compensation:
                            speed_ms_relative, camera_speed_ms = self.compensate_camera_motion(
                                speed_ms_absolute, camera_motion, fps, self.pixels_per_meter
                            )
                            speed_kmh_relative = speed_ms_relative * 3.6
                        else:
                            speed_ms_relative = speed_ms_absolute
                            speed_kmh_relative = speed_kmh_absolute
                        
                        if speed_ms_absolute < 50:
                            self.track_stats['track_speeds_absolute'][track_id].append(speed_ms_absolute)
                            self.track_stats['track_speeds_relative'][track_id].append(speed_ms_relative)
                            self.track_stats['avg_speeds_absolute'][track_id].append(speed_ms_absolute)
                            self.track_stats['avg_speeds_relative'][track_id].append(speed_ms_relative)
                            
                            if speed_ms_absolute > self.track_stats['max_speeds_absolute'].get(track_id, 0):
                                self.track_stats['max_speeds_absolute'][track_id] = speed_ms_absolute
                            if speed_ms_relative > self.track_stats['max_speeds_relative'].get(track_id, 0):
                                self.track_stats['max_speeds_relative'][track_id] = speed_ms_relative
                    
                    self.track_stats['track_history'][track_id].append(center)
                    if len(self.track_stats['track_history'][track_id]) > 100:
                        self.track_stats['track_history'][track_id].pop(0)
                
                frame_stats[class_label].append({
                    'track_id': track_id,
                    'confidence': round(conf, 3),
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'speed_ms_absolute': round(speed_ms_absolute, 2),
                    'speed_kmh_absolute': round(speed_kmh_absolute, 2),
                    'speed_ms_relative': round(speed_ms_relative, 2),
                    'speed_kmh_relative': round(speed_kmh_relative, 2),
                    'camera_speed_ms': round(camera_speed_ms, 2),
                    'center': center
                })
                
                color = self.COLORS.get(class_label, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if track_id is not None:
                    if self.enable_camera_compensation:
                        label_text = f"{class_label} (ID:{track_id}) {speed_kmh_relative:.1f}km/h"
                    else:
                        label_text = f"{class_label} (ID:{track_id}) {speed_kmh_absolute:.1f}km/h"
                else:
                    label_text = f"{class_label}: {conf:.2f}"
                
                label_size, baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - baseline - 5),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                if track_id is not None:
                    points = self.track_stats['track_history'][track_id]
                    if len(points) > 1:
                        for i in range(1, len(points)):
                            cv2.line(frame, points[i-1], points[i], color, 2)
                    cv2.circle(frame, center, 4, color, -1)
        
        return frame, frame_stats
    
    def draw_statistics(self, frame, frame_stats, unique_stats, frame_number, fps, camera_motion, congestion_level, congestion_details):
        """在帧上绘制统计信息和拥堵等级"""
        h, w = frame.shape[:2]
        
        # 获取拥堵等级颜色
        congestion_color = self.CONGESTION_COLORS.get(congestion_level, (255, 255, 255))
        congestion_name_cn = self.CONGESTION_NAMES_CN.get(congestion_level, '未知')
        congestion_name_en = self.CONGESTION_NAMES_EN.get(congestion_level, 'Unknown')
        
        # 1. 左上角：拥堵等级大显示
        if self.enable_congestion_detection:
            overlay_congestion = frame.copy()
            cv2.rectangle(overlay_congestion, (10, 10), (400, 100), congestion_color, -1)
            cv2.addWeighted(overlay_congestion, 0.3, frame, 0.7, 0, frame)
            
            text_color = (255, 255, 255)
            
            # 使用中文或英文显示拥堵等级
            if self.use_chinese_text:
                frame = self.put_chinese_text(frame, f"拥堵等级: {congestion_name_cn}", (20, 40), 24, text_color)
            else:
                cv2.putText(frame, f"Congestion: {congestion_name_en}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)
            
            level_text = f"Level: {congestion_details['congestion_score']:.2f}"
            cv2.putText(frame, level_text, (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # 2. 左中：检测信息
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 110), (450, 290), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        text_color = (255, 255, 255)
        y_offset = 135
        
        current_time = frame_number / fps
        time_text = f"Time: {current_time:.1f}s"
        cv2.putText(frame, time_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        camera_motion_pixels = math.sqrt(camera_motion[0]**2 + camera_motion[1]**2)
        camera_speed_ms, camera_speed_kmh = self.calculate_speed(
            camera_motion_pixels, fps, self.pixels_per_meter
        )
        camera_text = f"Camera: {camera_speed_kmh:.1f}km/h"
        cv2.putText(frame, camera_text, (20, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        
        y_offset += 50
        cv2.putText(frame, "Objects Detected:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        y_offset += 25
        cv2.putText(frame, f"Pedestrian: {len(frame_stats['pedestrian'])}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['pedestrian'], 2)
        
        y_offset += 20
        cv2.putText(frame, f"Bicycle: {len(frame_stats['bicycle'])}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['bicycle'], 2)
        
        y_offset += 20
        cv2.putText(frame, f"Car: {len(frame_stats['car'])}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['car'], 2)
        
        y_offset += 20
        cv2.putText(frame, f"Bus: {len(frame_stats['bus'])}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['bus'], 2)
        
        # 3. 右上角：总体统计
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (w - 500, 10), (w - 10, 290), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.3, frame, 0.7, 0, frame)
        
        y_offset = 35
        cv2.putText(frame, "Total Unique Count:", (w - 490, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        y_offset += 25
        cv2.putText(frame, f"Pedestrian: {len(unique_stats['pedestrians'])}", (w - 490, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['pedestrian'], 2)
        
        y_offset += 20
        cv2.putText(frame, f"Bicycle: {len(unique_stats['bicycles'])}", (w - 490, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['bicycle'], 2)
        
        y_offset += 20
        cv2.putText(frame, f"Car: {len(unique_stats['cars'])}", (w - 490, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['car'], 2)
        
        y_offset += 20
        cv2.putText(frame, f"Bus: {len(unique_stats['buses'])}", (w - 490, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['bus'], 2)
        
        # 拥堵因素分析
        y_offset += 30
        if self.use_chinese_text:
            frame = self.put_chinese_text(frame, "拥堵因素:", (w - 490, y_offset), 16, congestion_color)
        else:
            cv2.putText(frame, "Congestion Factors:", (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, congestion_color, 2)
        
        y_offset += 25
        for i, factor in enumerate(congestion_details['congestion_factors'][:3]):
            if self.use_chinese_text:
                frame = self.put_chinese_text(frame, f"• {factor}", (w - 480, y_offset + i*20), 12, congestion_color)
            else:
                # 将中文因素翻译为英文
                factor_en = {
                    '极低速度': 'Very Low Speed',
                    '低速度': 'Low Speed',
                    '中等速度': 'Medium Speed',
                    '速度急剧下降': 'Sharp Speed Drop',
                    '速度显著下降': 'Significant Speed Drop',
                    '速度略有下降': 'Slight Speed Drop',
                    '交通密度极高': 'Extreme Density',
                    '交通密度很高': 'High Density',
                    '交通密度较高': 'Medium Density'
                }.get(factor, factor)
                cv2.putText(frame, f"• {factor_en}", (w - 480, y_offset + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, congestion_color, 1)
        
        # 4. 下方：速度和密度信息
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (10, h - 120), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay3, 0.3, frame, 0.7, 0, frame)
        
        y_offset = h - 90
        if self.enable_camera_compensation:
            if self.use_chinese_text:
                frame = self.put_chinese_text(frame, "相对速度 (km/h) [已补偿摄像头运动]:", (20, y_offset), 16, text_color)
            else:
                cv2.putText(frame, "Relative Speed (km/h) [Camera Compensated]:", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        else:
            cv2.putText(frame, "Absolute Speed (km/h):", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        y_offset += 25
        speed_info = []
        
        # 显示每个类别的平均速度
        for class_label, speeds in self.track_stats['track_speeds_relative'].items():
            if speeds and self.enable_camera_compensation:
                avg_speed = np.mean(speeds[-5:]) * 3.6  # 最近5帧的平均相对速度
                speed_info.append(f"{class_label}: {avg_speed:.1f}")
            elif speeds:
                avg_speed = np.mean(speeds[-5:]) * 3.6  # 最近5帧的平均绝对速度
                speed_info.append(f"{class_label}: {avg_speed:.1f}")
        
        if speed_info:
            speed_text = " | ".join(speed_info)
        else:
            speed_text = "Waiting for tracking data..."
        
        cv2.putText(frame, speed_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def detect_and_track_video(self, video_path, save_output=True):
        """检测并追踪视频中的对象"""
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return None, None
        
        logger.info(f"开始处理视频: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area = width * height
        
        logger.info(f"视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧")
        
        video_name = Path(video_path).stem
        output_path = os.path.join(
            self.output_dir,
            f"{video_name}_tracked_congestion.mp4"
        )
        
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error("无法创建输出视频写入器")
                cap.release()
                return None, None
        
        self.track_stats = {
            'total_frames': 0,
            'unique_pedestrians': set(),
            'unique_bicycles': set(),
            'unique_cars': set(),
            'unique_buses': set(),
            'track_history': defaultdict(list),
            'track_speeds_absolute': defaultdict(list),
            'track_speeds_relative': defaultdict(list),
            'frame_detections': [],
            'max_speeds_absolute': defaultdict(float),
            'max_speeds_relative': defaultdict(float),
            'avg_speeds_absolute': defaultdict(list),
            'avg_speeds_relative': defaultdict(list),
            'congestion_history': [],
        }
        
        self.camera_stats = {
            'motions': [],
            'total_motion': (0, 0),
        }
        
        frame_count = 0
        
        logger.info("开始检测和追踪...")
        if self.enable_camera_compensation:
            logger.info("✅ 摄像头运动补偿已启用")
        if self.enable_congestion_detection:
            logger.info("✅ 拥堵检测已启用")
        if self.use_chinese_text:
            logger.info("✅ 使用中文显示")
        else:
            logger.info("✅ 使用英文显示")
        
        with tqdm(total=total_frames, desc="处理进度") as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 检测摄像头运动
                camera_motion = self.camera_motion_detector.detect_motion(frame)
                self.camera_stats['motions'].append(camera_motion)
                
                # YOLOv8检测 + ByteTrack追踪
                try:
                    results = self.model.track(
                        frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        tracker=self.tracker_type,
                        verbose=False,
                        persist=True
                    )
                except Exception as e:
                    logger.warning(f"追踪处理失败: {e}")
                    # 如果追踪失败，使用普通检测
                    results = self.model(
                        frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False
                    )
                
                # 绘制检测结果
                frame, frame_stats = self.draw_detections_with_tracking(
                    frame, results, fps, camera_motion
                )
                
                # 更新统计
                self.track_stats['total_frames'] += 1
                
                self.track_stats['frame_detections'].append({
                    'frame': frame_count,
                    'detections': frame_stats['detections'],
                    'camera_motion': camera_motion,
                })
                
                # 收集速度数据用于拥堵检测
                speeds_for_congestion = []
                for track_id in list(self.track_stats['track_speeds_relative'].keys()):
                    speeds = self.track_stats['track_speeds_relative'][track_id]
                    if speeds:
                        speeds_for_congestion.extend(speeds[-5:])  # 最近5个速度
                
                # 检测拥堵等级
                current_object_count = (len(frame_stats['pedestrian']) + 
                                       len(frame_stats['bicycle']) +
                                       len(frame_stats['car']) +
                                       len(frame_stats['bus']))
                
                congestion_level, congestion_details = self.congestion_detector.detect_congestion_level(
                    speeds_for_congestion, 
                    current_object_count,
                    frame_area
                )
                
                self.track_stats['congestion_history'].append({
                    'frame': frame_count,
                    'level': congestion_level.value,
                    'details': {k: v for k, v in congestion_details.items() if k != 'congestion_factors'}
                })
                
                # 绘制统计和拥堵信息
                unique_stats = {
                    'pedestrians': self.track_stats['unique_pedestrians'],
                    'bicycles': self.track_stats['unique_bicycles'],
                    'cars': self.track_stats['unique_cars'],
                    'buses': self.track_stats['unique_buses'],
                }
                frame = self.draw_statistics(frame, frame_stats, unique_stats, frame_count, fps, 
                                            camera_motion, congestion_level, congestion_details)
                
                # 写入输出视频
                if save_output:
                    out.write(frame)
                
                frame_count += 1
                pbar.update(1)
        
        # 释放资源
        cap.release()
        if save_output:
            out.release()
        
        logger.info(f"✅ 处理完成: {output_path}")
        
        return self.track_stats, output_path
    
    def save_statistics(self, stats, video_name):
        """保存统计结果"""
        stats_file = os.path.join(
            self.output_dir,
            'stats',
            f"{video_name}_stats.json"
        )
        
        # 计算每个对象的平均速度
        object_speeds = {}
        
        for track_id in list(stats['avg_speeds_relative'].keys()) + list(stats['avg_speeds_absolute'].keys()):
            if track_id in stats['avg_speeds_relative'] and stats['avg_speeds_relative'][track_id]:
                speeds_relative = stats['avg_speeds_relative'][track_id]
                avg_speed_ms_relative = np.mean(speeds_relative)
                max_speed_ms_relative = np.max(speeds_relative)
            else:
                avg_speed_ms_relative = 0
                max_speed_ms_relative = 0
            
            if track_id in stats['avg_speeds_absolute'] and stats['avg_speeds_absolute'][track_id]:
                speeds_absolute = stats['avg_speeds_absolute'][track_id]
                avg_speed_ms_absolute = np.mean(speeds_absolute)
                max_speed_ms_absolute = np.max(speeds_absolute)
            else:
                avg_speed_ms_absolute = 0
                max_speed_ms_absolute = 0
            
            obj_type = 'unknown'
            if track_id in stats['unique_pedestrians']:
                obj_type = 'pedestrian'
            elif track_id in stats['unique_bicycles']:
                obj_type = 'bicycle'
            elif track_id in stats['unique_cars']:
                obj_type = 'car'
            elif track_id in stats['unique_buses']:
                obj_type = 'bus'
            
            if avg_speed_ms_relative > 0 or avg_speed_ms_absolute > 0:
                object_speeds[f"{obj_type}_ID_{track_id}"] = {
                    'type': obj_type,
                    'track_id': track_id,
                    'absolute_speed': {
                        'avg_ms': round(avg_speed_ms_absolute, 3),
                        'avg_kmh': round(avg_speed_ms_absolute * 3.6, 2),
                        'max_ms': round(max_speed_ms_absolute, 3),
                        'max_kmh': round(max_speed_ms_absolute * 3.6, 2),
                    },
                    'relative_speed': {
                        'avg_ms': round(avg_speed_ms_relative, 3),
                        'avg_kmh': round(avg_speed_ms_relative * 3.6, 2),
                        'max_ms': round(max_speed_ms_relative, 3),
                        'max_kmh': round(max_speed_ms_relative * 3.6, 2),
                    },
                    'frames_tracked': len(stats['avg_speeds_relative'].get(track_id, [])),
                }
        
        # 计算每个类别的平均速度
        category_speeds = {}
        for class_type in ['pedestrian', 'bicycle', 'car', 'bus']:
            all_speeds_relative = []
            all_speeds_absolute = []
            track_ids = getattr(stats, f'unique_{class_type}s', set())
            
            for track_id in track_ids:
                if track_id in stats['avg_speeds_relative'] and stats['avg_speeds_relative'][track_id]:
                    all_speeds_relative.extend(stats['avg_speeds_relative'][track_id])
                if track_id in stats['avg_speeds_absolute'] and stats['avg_speeds_absolute'][track_id]:
                    all_speeds_absolute.extend(stats['avg_speeds_absolute'][track_id])
            
            if all_speeds_relative or all_speeds_absolute:
                category_speeds[class_type] = {
                    'count': len(track_ids),
                    'absolute_speed': {
                        'avg_ms': round(np.mean(all_speeds_absolute), 3) if all_speeds_absolute else 0,
                        'avg_kmh': round(np.mean(all_speeds_absolute) * 3.6, 2) if all_speeds_absolute else 0,
                        'max_ms': round(np.max(all_speeds_absolute), 3) if all_speeds_absolute else 0,
                        'max_kmh': round(np.max(all_speeds_absolute) * 3.6, 2) if all_speeds_absolute else 0,
                    },
                    'relative_speed': {
                        'avg_ms': round(np.mean(all_speeds_relative), 3) if all_speeds_relative else 0,
                        'avg_kmh': round(np.mean(all_speeds_relative) * 3.6, 2) if all_speeds_relative else 0,
                        'max_ms': round(np.max(all_speeds_relative), 3) if all_speeds_relative else 0,
                        'max_kmh': round(np.max(all_speeds_relative) * 3.6, 2) if all_speeds_relative else 0,
                    }
                }
        
        # 计算摄像头运动统计
        if self.camera_stats['motions']:
            camera_motions_pixels = [math.sqrt(m[0]**2 + m[1]**2) for m in self.camera_stats['motions']]
            avg_camera_motion_pixels = np.mean(camera_motions_pixels)
            max_camera_motion_pixels = np.max(camera_motions_pixels)
            
            avg_camera_speed_ms, avg_camera_speed_kmh = self.calculate_speed(
                avg_camera_motion_pixels, len(self.camera_stats['motions']), self.pixels_per_meter
            )
            max_camera_speed_ms, max_camera_speed_kmh = self.calculate_speed(
                max_camera_motion_pixels, len(self.camera_stats['motions']), self.pixels_per_meter
            )
            
            camera_speed_stats = {
                'avg_speed_ms': round(avg_camera_speed_ms, 3),
                'avg_speed_kmh': round(avg_camera_speed_kmh, 2),
                'max_speed_ms': round(max_camera_speed_ms, 3),
                'max_speed_kmh': round(max_camera_speed_kmh, 2),
                'has_motion': max_camera_speed_kmh > 0.5,
            }
        else:
            camera_speed_stats = {'has_motion': False}
        
        # 计算拥堵统计
        if stats['congestion_history']:
            congestion_counts = defaultdict(int)
            for cong_data in stats['congestion_history']:
                level = cong_data['level']
                congestion_counts[level] += 1
            
            total_congestion_frames = len(stats['congestion_history'])
            congestion_stats = {
                'total_frames_analyzed': total_congestion_frames,
                'level_distribution': {k: v for k, v in congestion_counts.items()},
                'level_percentages': {k: round(100 * v / total_congestion_frames, 2) 
                                      for k, v in congestion_counts.items()},
                'most_common_level': max(congestion_counts, key=congestion_counts.get) if congestion_counts else 'unknown'
            }
        else:
            congestion_stats = {'total_frames_analyzed': 0}
        
        result_stats = {
            'video': video_name,
            'timestamp': datetime.now().isoformat(),
            'total_frames': stats['total_frames'],
            'unique_counts': {
                'pedestrian': len(stats['unique_pedestrians']),
                'bicycle': len(stats['unique_bicycles']),
                'car': len(stats['unique_cars']),
                'bus': len(stats['unique_buses']),
                'total': (len(stats['unique_pedestrians']) + 
                         len(stats['unique_bicycles']) +
                         len(stats['unique_cars']) +
                         len(stats['unique_buses']))
            },
            'speed_statistics': {
                'by_category': category_speeds,
                'by_object': object_speeds,
            },
            'camera_motion': camera_speed_stats,
            'congestion_analysis': congestion_stats,
            'congestion_levels': {
                'smooth': '畅通',
                'light': '轻微拥堵',
                'moderate': '中等拥堵',
                'heavy': '严重拥堵',
                'severe': '极度拥堵',
            },
            'model': self.model_name,
            'confidence_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'tracker': self.tracker_type,
            'camera_compensation': self.enable_camera_compensation,
            'congestion_detection': self.enable_congestion_detection,
            'chinese_text': self.use_chinese_text,
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(result_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"统计结果已保存: {stats_file}")
        
        return stats_file
    
    def print_statistics(self, stats, video_name):
        """打印统计结果"""
        print("\n" + "=" * 100)
        print(f"📊 检测追踪统计结果 - {video_name}")
        print("=" * 100)
        print(f"总帧数: {stats['total_frames']}")
        
        print(f"\n🚶 行人 (Pedestrian):")
        print(f"   唯一ID数: {len(stats['unique_pedestrians'])}")
        
        print(f"\n🚴 自行车 (Bicycle):")
        print(f"   唯一ID数: {len(stats['unique_bicycles'])}")
        
        print(f"\n🚗 小轿车 (Car):")
        print(f"   唯一ID数: {len(stats['unique_cars'])}")
        
        print(f"\n🚌 公共汽车 (Bus):")
        print(f"   唯一ID数: {len(stats['unique_buses'])}")
        
        # 摄像头运动统计
        if self.camera_stats['motions']:
            camera_motions_pixels = [math.sqrt(m[0]**2 + m[1]**2) for m in self.camera_stats['motions']]
            avg_motion = np.mean(camera_motions_pixels)
            avg_speed_ms, avg_speed_kmh = self.calculate_speed(
                avg_motion, len(self.camera_stats['motions']), self.pixels_per_meter
            )
            print(f"\n📹 摄像头运动统计:")
            print(f"   平均运动速度: {avg_speed_kmh:.1f} km/h")
        
        # 拥堵统计
        if stats['congestion_history']:
            congestion_counts = defaultdict(int)
            for cong_data in stats['congestion_history']:
                level = cong_data['level']
                congestion_counts[level] += 1
            
            print(f"\n🚦 道路拥堵分析:")
            level_names = {
                'smooth': '畅通 ✅',
                'light': '轻微拥堵 ⚠️',
                'moderate': '中等拥堵 ⚠️⚠️',
                'heavy': '严重拥堵 🔴',
                'severe': '极度拥堵 🔴🔴',
            }
            
            for level, count in sorted(congestion_counts.items()):
                percentage = 100 * count / len(stats['congestion_history'])
                level_name = level_names.get(level, level)
                print(f"   {level_name}: {count}帧 ({percentage:.1f}%)")
         
        total = (len(stats['unique_pedestrians']) + 
                len(stats['unique_bicycles']) +
                len(stats['unique_cars']) +
                len(stats['unique_buses']))
        print(f"\n✅ 总唯一对象数: {total}")
        print("=" * 100 + "\n")
    
    def process_directory(self, directory=None, extensions=['.mp4', '.avi', '.mov', '.mkv']):
        """批量处理目录中的所有视频"""
        if directory is None:
            directory = self.input_dir
        
        video_files = []
        for ext in extensions:
            video_files.extend(Path(directory).glob(f"*{ext}"))
            video_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        if not video_files:
            logger.warning(f"未找到视频文件: {directory}")
            return []
        
        logger.info(f"找到 {len(video_files)} 个视频文件")
        
        results = []
        for idx, video_path in enumerate(video_files, 1):
            logger.info(f"\n[{idx}/{len(video_files)}] 处理: {video_path.name}")
            try:
                stats, output_path = self.detect_and_track_video(str(video_path))
                
                if stats and output_path:
                    self.print_statistics(stats, video_path.stem)
                    self.save_statistics(stats, video_path.stem)
                    
                    results.append({
                        'input': str(video_path),
                        'output': output_path,
                        'stats': stats
                    })
            except Exception as e:
                logger.error(f"处理视频失败: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\n✅ 总共成功处理 {len(results)}/{len(video_files)} 个视频")
        
        return results
    
    def generate_summary_report(self, results):
        """生成总结报告"""
        if not results:
            logger.warning("没有处理结果")
            return
        
        report_file = os.path.join(self.output_dir, 'detection_summary.json')
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': len(results),
            'tracking_method': self.tracker_type,
            'model': self.model_name,
            'camera_compensation_enabled': self.enable_camera_compensation,
            'congestion_detection_enabled': self.enable_congestion_detection,
            'chinese_text_enabled': self.use_chinese_text,
            'total_unique_counts': {
                'pedestrian': sum(len(r['stats']['unique_pedestrians']) for r in results),
                'bicycle': sum(len(r['stats']['unique_bicycles']) for r in results),
                'car': sum(len(r['stats']['unique_cars']) for r in results),
                'bus': sum(len(r['stats']['unique_buses']) for r in results),
            },
            'videos': []
        }
        
        for result in results:
            summary['videos'].append({
                'name': Path(result['input']).stem,
                'input': result['input'],
                'output': result['output'],
                'unique_counts': {
                    'pedestrian': len(result['stats']['unique_pedestrians']),
                    'bicycle': len(result['stats']['unique_bicycles']),
                    'car': len(result['stats']['unique_cars']),
                    'bus': len(result['stats']['unique_buses']),
                }
            })
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"总结报告已保存: {report_file}")
        
        print("\n" + "=" * 100)
        print("📈 整体追踪总结")
        print("=" * 100)
        print(f"处理视频数: {summary['total_videos']}")
        print(f"追踪方法: {summary['tracking_method']}")
        print(f"摄像头运动补偿: {'✅ 启用' if summary['camera_compensation_enabled'] else '❌ 禁用'}")
        print(f"拥堵检测: {'✅ 启用' if summary['congestion_detection_enabled'] else '❌ 禁用'}")
        print(f"中文显示: {'✅ 启用' if summary['chinese_text_enabled'] else '❌ 禁用'}")
        print(f"\n全部视频累计唯一对象数:")
        print(f"  行人: {summary['total_unique_counts']['pedestrian']}")
        print(f"  自行车: {summary['total_unique_counts']['bicycle']}")
        print(f"  小轿车: {summary['total_unique_counts']['car']}")
        print(f"  公共汽车: {summary['total_unique_counts']['bus']}")
        
        total = sum(summary['total_unique_counts'].values())
        print(f"\n总计: {total}")
        print("=" * 100 + "\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='YOLOv8 + ByteTrack 检测追踪工具（带拥堵检测）'
    )
    parser.add_argument('input', nargs='?', default='pre_video',
                       help='输入视频文件或目录')
    parser.add_argument('-o', '--output', default='output_video_2',
                       help='输出目录')
    parser.add_argument('-m', '--model', default='yolov8l.pt',
                       help='YOLOv8模型')
    parser.add_argument('-c', '--conf', type=float, default=0.3,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='NMS IOU阈值')
    parser.add_argument('-t', '--tracker', default='bytetrack.yaml',
                       help='追踪器类型配置文件')
    parser.add_argument('--ppm', type=float, default=50,
                       help='像素与米的转换比例')
    parser.add_argument('--no-camera-compensation', action='store_true',
                       help='禁用摄像头运动补偿')
    parser.add_argument('--no-congestion-detection', action='store_true',
                       help='禁用拥堵检测')
    parser.add_argument('--no-chinese', action='store_true',
                       help='禁用中文显示（使用英文）')
    
    args = parser.parse_args()
    
    detector = YOLOTrackerDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output,
        tracker=args.tracker,
        pixels_per_meter=args.ppm,
        enable_camera_compensation=not args.no_camera_compensation,
        enable_congestion_detection=not args.no_congestion_detection,
        use_chinese_text=not args.no_chinese
    )
    
    if os.path.isfile(args.input):
        logger.info(f"处理单个视频文件: {args.input}")
        stats, output_path = detector.detect_and_track_video(args.input)
        if stats:
            detector.print_statistics(stats, Path(args.input).stem)
            detector.save_statistics(stats, Path(args.input).stem)
    elif os.path.isdir(args.input):
        logger.info(f"批量处理目录: {args.input}")
        results = detector.process_directory(args.input)
        if results:
            detector.generate_summary_report(results)
    else:
        logger.error(f"无效的输入路径: {args.input}")


if __name__ == "__main__":
    main()
