#!/usr/bin/env python3
"""
WildGS-SLAM 自定义数据集处理脚本
支持视频转图像、相机标定、数据格式转换等功能
"""

import os
import sys
import cv2
import numpy as np
import argparse
import json
import yaml
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging
from dataclasses import dataclass
import shutil

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CameraIntrinsics:
    """相机内参数据类"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0  # 径向畸变参数
    k2: float = 0.0
    p1: float = 0.0  # 切向畸变参数
    p2: float = 0.0

class CustomDataProcessor:
    """自定义数据集处理器"""
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建输出子目录
        self.images_dir = self.output_path / "rgb"
        self.images_dir.mkdir(exist_ok=True)
        
        logger.info(f"输入路径: {self.input_path}")
        logger.info(f"输出路径: {self.output_path}")
    
    def extract_frames_from_video(self, video_path: str, fps: Optional[float] = None,
                                 start_time: float = 0, end_time: Optional[float] = None,
                                 max_frames: Optional[int] = None) -> bool:
        """从视频提取帧"""
        logger.info(f"从视频提取帧: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return False
        
        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        logger.info(f"视频信息: {video_fps:.2f} FPS, {total_frames} 帧, {duration:.2f} 秒")
        
        # 设置提取参数
        if fps is None:
            fps = video_fps
        
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps) if end_time else total_frames
        
        logger.info(f"提取设置: 每 {frame_interval} 帧提取一帧")
        logger.info(f"提取范围: 第 {start_frame} 到 {end_frame} 帧")
        
        # 跳转到开始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and extracted_count >= max_frames):
                break
            
            current_frame = start_frame + frame_count
            if current_frame >= end_frame:
                break
            
            # 按间隔提取帧
            if frame_count % frame_interval == 0:
                # 保存帧
                frame_filename = f"{extracted_count:06d}.jpg"
                frame_path = self.images_dir / frame_filename
                
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1
                
                if extracted_count % 100 == 0:
                    logger.info(f"已提取 {extracted_count} 帧")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"帧提取完成: 共提取 {extracted_count} 帧")
        
        # 保存时间戳文件
        timestamps_path = self.output_path / "timestamps.txt"
        with open(timestamps_path, 'w') as f:
            for i in range(extracted_count):
                timestamp = start_time + (i * frame_interval / video_fps)
                f.write(f"{timestamp:.6f}\n")
        
        return True
    
    def process_image_sequence(self, images_path: str, 
                              image_pattern: str = "*.jpg",
                              resize_resolution: Optional[Tuple[int, int]] = None) -> bool:
        """处理图像序列"""
        logger.info(f"处理图像序列: {images_path}")
        
        images_dir = Path(images_path)
        if not images_dir.exists():
            logger.error(f"图像目录不存在: {images_dir}")
            return False
        
        # 获取所有图像文件
        image_files = sorted(list(images_dir.glob(image_pattern)))
        if not image_files:
            logger.error(f"未找到匹配的图像文件: {image_pattern}")
            return False
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        # 处理每个图像
        for i, img_path in enumerate(image_files):
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"无法读取图像: {img_path}")
                continue
            
            # 调整分辨率
            if resize_resolution:
                img = cv2.resize(img, resize_resolution)
            
            # 保存图像
            output_filename = f"{i:06d}.jpg"
            output_path = self.images_dir / output_filename
            cv2.imwrite(str(output_path), img)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1} 个图像")
        
        logger.info(f"图像序列处理完成: 共处理 {len(image_files)} 个图像")
        return True
    
    def calibrate_camera_with_checkerboard(self, calibration_images_path: str,
                                          checkerboard_size: Tuple[int, int] = (9, 6),
                                          square_size: float = 1.0) -> Optional[CameraIntrinsics]:
        """使用棋盘格标定相机"""
        logger.info("开始相机标定...")
        
        calib_dir = Path(calibration_images_path)
        if not calib_dir.exists():
            logger.error(f"标定图像目录不存在: {calib_dir}")
            return None
        
        # 获取标定图像
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(calib_dir.glob(ext))
        
        if len(image_files) < 10:
            logger.error("标定图像数量不足（至少需要10张）")
            return None
        
        logger.info(f"找到 {len(image_files)} 张标定图像")
        
        # 准备标定点
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 3D点
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # 存储3D点和2D点
        objpoints = []  # 3D点
        imgpoints = []  # 2D点
        
        img_shape = None
        successful_images = 0
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_shape = gray.shape[::-1]
            
            # 寻找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # 亚像素精度角点
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                successful_images += 1
                logger.info(f"成功检测到角点: {img_path.name}")
            else:
                logger.warning(f"未能检测到角点: {img_path.name}")
        
        if successful_images < 10:
            logger.error(f"成功检测角点的图像不足: {successful_images}/10")
            return None
        
        logger.info(f"用于标定的图像数量: {successful_images}")
        
        # 执行相机标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
        
        if not ret:
            logger.error("相机标定失败")
            return None
        
        # 计算重投影误差
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        mean_error /= len(objpoints)
        logger.info(f"平均重投影误差: {mean_error:.3f} 像素")
        
        # 创建相机内参对象
        intrinsics = CameraIntrinsics(
            fx=mtx[0, 0],
            fy=mtx[1, 1],
            cx=mtx[0, 2],
            cy=mtx[1, 2],
            width=img_shape[0],
            height=img_shape[1],
            k1=dist[0, 0],
            k2=dist[0, 1],
            p1=dist[0, 2],
            p2=dist[0, 3]
        )
        
        logger.info("相机标定完成")
        logger.info(f"内参: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
        logger.info(f"主点: cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}")
        logger.info(f"分辨率: {intrinsics.width}x{intrinsics.height}")
        
        return intrinsics
    
    def save_camera_intrinsics(self, intrinsics: CameraIntrinsics, format_type: str = "wildgs"):
        """保存相机内参"""
        if format_type == "wildgs":
            # WildGS-SLAM格式
            camera_info = {
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.cx,
                'cy': intrinsics.cy,
                'width': intrinsics.width,
                'height': intrinsics.height,
                'k1': intrinsics.k1,
                'k2': intrinsics.k2,
                'p1': intrinsics.p1,
                'p2': intrinsics.p2
            }
            
            with open(self.output_path / "camera_intrinsics.json", 'w') as f:
                json.dump(camera_info, f, indent=2)
        
        elif format_type == "tum":
            # TUM格式
            with open(self.output_path / "camera.txt", 'w') as f:
                f.write(f"{intrinsics.fx} {intrinsics.fy} {intrinsics.cx} {intrinsics.cy}\n")
        
        elif format_type == "colmap":
            # COLMAP格式
            with open(self.output_path / "cameras.txt", 'w') as f:
                f.write("# Camera list with one line of data per camera:\n")
                f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
                f.write(f"1 PINHOLE {intrinsics.width} {intrinsics.height} ")
                f.write(f"{intrinsics.fx} {intrinsics.fy} {intrinsics.cx} {intrinsics.cy}\n")
        
        logger.info(f"相机内参已保存为 {format_type} 格式")
    
    def generate_wildgs_config(self, 
                              intrinsics: CameraIntrinsics,
                              sequence_name: str = "custom_sequence",
                              template_config: Optional[str] = None) -> str:
        """生成WildGS-SLAM配置文件"""
        
        if template_config and Path(template_config).exists():
            # 从模板加载配置
            with open(template_config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # 使用默认配置
            config = {
                'scene': sequence_name,
                'input_folder': str(self.output_path),
                'output_folder': str(self.output_path / "results"),
                'cam': {
                    'H': intrinsics.height,
                    'W': intrinsics.width,
                    'H_out': intrinsics.height,
                    'W_out': intrinsics.width,
                    'fx': intrinsics.fx,
                    'fy': intrinsics.fy,
                    'cx': intrinsics.cx,
                    'cy': intrinsics.cy,
                },
                'tracking': {
                    'use_depth_filter': True,
                    'depth_filter_thresh': 0.2,
                    'keyframe_interval': 5,
                    'uncertainty_threshold': 0.8
                },
                'mapping': {
                    'gaussians_start': 1000,
                    'gaussians_end': 500000,
                    'pruning_interval': 100,
                    'gaussian_lr': 0.01,
                    'position_lr': 0.00016,
                    'feature_lr': 0.0025
                }
            }
        
        # 更新相机参数
        config['cam'].update({
            'H': intrinsics.height,
            'W': intrinsics.width,
            'H_out': intrinsics.height,
            'W_out': intrinsics.width,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'cx': intrinsics.cx,
            'cy': intrinsics.cy,
        })
        
        # 保存配置文件
        config_path = self.output_path / f"{sequence_name}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"WildGS-SLAM配置文件已保存: {config_path}")
        return str(config_path)
    
    def create_dataset_structure(self, intrinsics: CameraIntrinsics, 
                               sequence_name: str = "custom_sequence"):
        """创建完整的数据集结构"""
        logger.info("创建数据集结构...")
        
        # 创建必要的目录结构
        (self.output_path / "results").mkdir(exist_ok=True)
        
        # 保存多种格式的相机内参
        self.save_camera_intrinsics(intrinsics, "wildgs")
        self.save_camera_intrinsics(intrinsics, "tum")
        self.save_camera_intrinsics(intrinsics, "colmap")
        
        # 生成配置文件
        config_path = self.generate_wildgs_config(intrinsics, sequence_name)
        
        # 创建README文件
        readme_content = f"""# {sequence_name} Dataset

## 数据集信息
- 序列名称: {sequence_name}
- 图像数量: {len(list(self.images_dir.glob('*.jpg')))}
- 图像分辨率: {intrinsics.width}x{intrinsics.height}
- 相机内参: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}

## 目录结构
```
{self.output_path.name}/
├── rgb/                    # RGB图像
├── camera_intrinsics.json  # WildGS-SLAM格式相机内参
├── camera.txt             # TUM格式相机内参
├── cameras.txt            # COLMAP格式相机内参
├── timestamps.txt         # 时间戳（如果从视频提取）
├── {sequence_name}_config.yaml  # WildGS-SLAM配置文件
└── results/               # 运行结果目录
```

## 使用方法

### 运行WildGS-SLAM
```bash
cd WildGS-SLAM
python run.py {config_path}
```

### 评估结果
```bash
python scripts_run/summarize_pose_eval.py
```
"""
        
        with open(self.output_path / "README.md", 'w') as f:
            f.write(readme_content)
        
        logger.info("数据集结构创建完成")
        
        # 显示摘要信息
        self.print_dataset_summary(intrinsics, sequence_name)
    
    def print_dataset_summary(self, intrinsics: CameraIntrinsics, sequence_name: str):
        """打印数据集摘要"""
        num_images = len(list(self.images_dir.glob('*.jpg')))
        
        print("\n" + "="*60)
        print(f"数据集处理完成: {sequence_name}")
        print("="*60)
        print(f"输出目录: {self.output_path}")
        print(f"图像数量: {num_images}")
        print(f"图像分辨率: {intrinsics.width}x{intrinsics.height}")
        print(f"相机内参:")
        print(f"  fx: {intrinsics.fx:.2f}")
        print(f"  fy: {intrinsics.fy:.2f}")
        print(f"  cx: {intrinsics.cx:.2f}")
        print(f"  cy: {intrinsics.cy:.2f}")
        print(f"配置文件: {sequence_name}_config.yaml")
        print(f"README文件: README.md")
        print("="*60)
        print("下一步:")
        print(f"1. 查看数据集: cd {self.output_path}")
        print(f"2. 运行SLAM: cd WildGS-SLAM && python run.py {self.output_path}/{sequence_name}_config.yaml")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='WildGS-SLAM Custom Dataset Processor')
    parser.add_argument('input_path', help='Input path (video file or images directory)')
    parser.add_argument('output_path', help='Output dataset directory')
    parser.add_argument('--sequence-name', default='custom_sequence', help='Dataset sequence name')
    
    # 视频处理选项
    parser.add_argument('--from-video', action='store_true', help='Extract frames from video')
    parser.add_argument('--fps', type=float, help='Extraction FPS (default: same as video)')
    parser.add_argument('--start-time', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--end-time', type=float, help='End time in seconds')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to extract')
    
    # 图像处理选项
    parser.add_argument('--image-pattern', default='*.jpg', help='Image pattern for input images')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Resize images to specified resolution')
    
    # 相机标定选项
    parser.add_argument('--calibrate', action='store_true', help='Perform camera calibration')
    parser.add_argument('--calib-images', help='Path to calibration images')
    parser.add_argument('--checkerboard-size', nargs=2, type=int, default=[9, 6],
                       metavar=('WIDTH', 'HEIGHT'), help='Checkerboard size (corners)')
    parser.add_argument('--square-size', type=float, default=1.0, help='Checkerboard square size')
    
    # 手动相机参数
    parser.add_argument('--manual-intrinsics', nargs=6, type=float,
                       metavar=('FX', 'FY', 'CX', 'CY', 'WIDTH', 'HEIGHT'),
                       help='Manual camera intrinsics')
    
    # 配置选项
    parser.add_argument('--template-config', help='Template config file for WildGS-SLAM')
    parser.add_argument('--camera-format', choices=['wildgs', 'tum', 'colmap'], 
                       default='wildgs', help='Camera format to save')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = CustomDataProcessor(args.input_path, args.output_path)
    
    # 处理输入数据
    if args.from_video:
        success = processor.extract_frames_from_video(
            video_path=args.input_path,
            fps=args.fps,
            start_time=args.start_time,
            end_time=args.end_time,
            max_frames=args.max_frames
        )
        if not success:
            logger.error("视频帧提取失败")
            sys.exit(1)
    else:
        success = processor.process_image_sequence(
            images_path=args.input_path,
            image_pattern=args.image_pattern,
            resize_resolution=tuple(args.resize) if args.resize else None
        )
        if not success:
            logger.error("图像序列处理失败")
            sys.exit(1)
    
    # 获取相机内参
    intrinsics = None
    
    if args.calibrate:
        # 自动标定
        calib_path = args.calib_images or args.input_path
        intrinsics = processor.calibrate_camera_with_checkerboard(
            calibration_images_path=calib_path,
            checkerboard_size=tuple(args.checkerboard_size),
            square_size=args.square_size
        )
        if intrinsics is None:
            logger.error("相机标定失败")
            sys.exit(1)
    
    elif args.manual_intrinsics:
        # 手动设置内参
        fx, fy, cx, cy, width, height = args.manual_intrinsics
        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=int(width), height=int(height)
        )
        logger.info("使用手动设置的相机内参")
    
    else:
        # 使用默认内参（需要用户提供）
        logger.error("请提供相机内参：使用 --calibrate 进行自动标定或 --manual-intrinsics 手动设置")
        sys.exit(1)
    
    # 创建完整的数据集结构
    processor.create_dataset_structure(intrinsics, args.sequence_name)
    
    logger.info("数据集处理完成！")

if __name__ == '__main__':
    main()