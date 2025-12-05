"""
视频预处理程序
用于处理 .mp4 和 .mov 文件，为行人和车辆检测做准备
支持：格式转换、尺寸调整、帧率标准化、亮度对比度调整
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

# 配置日志
logging. basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """视频预处理类"""
    
    def __init__(self, 
                 output_dir='pre_video',
                 target_fps=30,
                 target_width=1280,
                 target_height=720,
                 codec='mp4v'):
        """
        初始化预处理器
        
        Args:
            output_dir (str): 输出目录
            target_fps (int): 目标帧率（默认30fps）
            target_width (int): 目标宽度（默认1280）
            target_height (int): 目标高度（默认720）
            codec (str): 输出编码格式（默认mp4v）
        """
        self. output_dir = output_dir
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.codec = codec
        
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {os.path.abspath(self. output_dir)}")
    
    def adjust_brightness_contrast (self, frame, brightness=0, contrast=1.0):
        """
        调整亮度和对比度
        
        Args:
            frame: 输入帧
            brightness: 亮度调整值 (-100 ~ 100)
            contrast: 对比度调整值 (0.5 ~ 3.0)
        
        Returns:
            调整后的帧
        """
        # 调整对比度
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        return frame
    
    def equalize_histogram(self, frame):
        """
        直方图均衡化，增强对比度
        
        Args:
            frame: 输入帧
        
        Returns:
            均衡化后的帧
        """
        # 分离通道
        if len(frame.shape) == 3:
            # 转换为HSV，只对V通道进行均衡化
            hsv = cv2.cvtColor(frame, cv2. COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            frame = cv2.equalizeHist(frame)
        
        return frame
    
    def denoise_frame(self, frame, method='bilateral'):
        """
        去噪处理
        
        Args:
            frame: 输入帧
            method: 去噪方法 ('bilateral', 'gaussian', 'morphological')
        
        Returns:
            去噪后的帧
        """
        if method == 'bilateral':
            # 双边滤波（保留边界）
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
        elif method == 'gaussian':
            # 高斯模糊
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
        elif method == 'morphological':
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            frame = cv2. morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        
        return frame
    
    def resize_frame(self, frame, width=None, height=None, maintain_aspect=True):
        """
        调整帧尺寸
        
        Args:
            frame: 输入帧
            width: 目标宽度
            height: 目标高度
            maintain_aspect: 是否保持纵横比
        
        Returns:
            调整后的帧
        """
        if width is None:
            width = self.target_width
        if height is None:
            height = self.target_height
        
        if maintain_aspect:
            # 计算缩放比例
            h, w = frame.shape[:2]
            scale = min(width / w, height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 调整大小
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 填充到目标尺寸
            canvas = np.zeros((height, width, 3), dtype=frame.dtype)
            y_offset = (height - new_h) // 2
            x_offset = (width - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = frame
            frame = canvas
        else:
            # 直接调整到目标尺寸
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        return frame
    
    def preprocess_frame(self, frame, denoise=True, equalize=True, adjust_brightness=True):
        """
        完整的帧预处理流程
        
        Args:
            frame: 输入帧
            denoise: 是否去噪
            equalize: 是否直方图均衡化
            adjust_brightness: 是否调整亮度对比度
        
        Returns:
            预处理后的帧
        """
        # 1. 调整尺寸
        frame = self.resize_frame(frame)
        
        # 2. 去噪
        if denoise:
            frame = self.denoise_frame(frame, method='bilateral')
        
        # 3. 直方图均衡化
        if equalize:
            frame = self. equalize_histogram(frame)
        
        # 4. 调整亮度对比度
        if adjust_brightness:
            frame = self.adjust_brightness_contrast(frame, brightness=10, contrast=1.1)
        
        return frame
    
    def get_video_info(self, video_path):
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            包含视频信息的字典
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        info = {
            'width': int(cap.get(cv2. CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2. CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def process_video(self, video_path, denoise=True, equalize=True, adjust_brightness=True):
        """
        处理单个视频文件
        
        Args:
            video_path: 输入视频路径
            denoise: 是否去噪
            equalize: 是否直方图均衡化
            adjust_brightness: 是否调整亮度对比度
        
        Returns:
            输出视频路径
        """
        # 检查文件是否存在
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return None
        
        # 获取视频信息
        try:
            info = self.get_video_info(video_path)
            logger.info(f"原始视频信息: {info['width']}x{info['height']}, "
                       f"{info['fps']:.2f}fps, {info['duration']:.2f}秒")
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return None
        
        # 打开输入视频
        cap = cv2.VideoCapture(video_path)
        
        # 生成输出文件名
        filename = Path(video_path).stem
        output_path = os.path.join(self.output_dir, f"{filename}_preprocessed.mp4")
        
        # 定义视频编写器
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2. VideoWriter(output_path, fourcc, self.target_fps, 
                             (self.target_width, self.target_height))
        
        if not out.isOpened():
            logger.error("无法创建输出视频写入器")
            cap.release()
            return None
        
        # 处理每一帧
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"开始处理视频: {os.path.basename(video_path)}")
        
        with tqdm(total=total_frames, desc="处理进度") as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 预处理帧
                processed_frame = self.preprocess_frame(
                    frame,
                    denoise=denoise,
                    equalize=equalize,
                    adjust_brightness=adjust_brightness
                )
                
                # 写入处理后的帧
                out.write(processed_frame)
                
                frame_count += 1
                pbar.update(1)
        
        # 释放资源
        cap.release()
        out.release()
        
        logger.info(f"视频处理完成: {output_path}")
        logger.info(f"处理帧数: {frame_count}, 目标帧率: {self.target_fps}fps")
        
        return output_path
    
    def process_directory(self, directory, extensions=['.mp4', '.mov'], 
                         denoise=True, equalize=True, adjust_brightness=True):
        """
        批量处理目录中的视频文件
        
        Args:
            directory: 输入目录
            extensions: 要处理的文件扩展名
            denoise: 是否去噪
            equalize: 是否直方图均衡化
            adjust_brightness: 是否调整亮度对比度
        
        Returns:
            处理成功的视频列表
        """
        video_files = []
        
        # 查找所有视频文件
        for ext in extensions:
            video_files.extend(Path(directory).glob(f"*{ext}"))
        
        if not video_files:
            logger.warning(f"未找到视频文件: {directory}")
            return []
        
        logger.info(f"找到 {len(video_files)} 个视频文件")
        
        processed_videos = []
        for idx, video_path in enumerate(video_files, 1):
            logger.info(f"\n[{idx}/{len(video_files)}] 处理: {video_path. name}")
            try:
                output_path = self.process_video(
                    str(video_path),
                    denoise=denoise,
                    equalize=equalize,
                    adjust_brightness=adjust_brightness
                )
                if output_path:
                    processed_videos.append(output_path)
            except Exception as e:
                logger.error(f"处理视频失败: {e}")
        
        logger.info(f"\n总共成功处理 {len(processed_videos)}/{len(video_files)} 个视频")
        return processed_videos


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='视频预处理工具')
    parser.add_argument('input', nargs='?', default='D:\\study_opencv\\new_project\\video',
                       help='输入视频文件或目录（默认当前目录）')
    parser.add_argument('-o', '--output', default='pre_video',
                       help='输出目录（默认pre_video）')
    parser. add_argument('-fps', '--target-fps', type=int, default=30,
                       help='目标帧率（默认30fps）')
    parser.add_argument('-w', '--width', type=int, default=1280,
                       help='目标宽度（默认1280）')
    parser.add_argument('-he', '--height', type=int, default=720,
                       help='目标高度（默认720）')
    parser.add_argument('--no-denoise', action='store_true',
                       help='禁用去噪')
    parser.add_argument('--no-equalize', action='store_true',
                       help='禁用直方图均衡化')
    parser.add_argument('--no-brightness-adjust', action='store_true',
                       help='禁用亮度对比度调整')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = VideoPreprocessor(
        output_dir=args.output,
        target_fps=args.target_fps,
        target_width=args.width,
        target_height=args.height
    )
    
    # 处理单个文件或目录
    if os.path.isfile(args. input):
        logger.info(f"处理单个视频文件: {args.input}")
        preprocessor.process_video(
            args.input,
            denoise=not args.no_denoise,
            equalize=not args.no_equalize,
            adjust_brightness=not args.no_brightness_adjust
        )
    elif os.path.isdir(args.input):
        logger.info(f"批量处理目录: {args.input}")
        preprocessor. process_directory(
            args. input,
            denoise=not args.no_denoise,
            equalize=not args.no_equalize,
            adjust_brightness=not args. no_brightness_adjust
        )
    else:
        logger.error(f"无效的输入路径: {args.input}")


if __name__ == "__main__":
    main()
