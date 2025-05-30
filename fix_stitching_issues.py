#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像拼接问题修复工具
解决图像加载失败和参数优化问题
"""

import cv2
import numpy as np
import os
import glob
from typing import List, Optional
import logging
from PIL import Image
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class ImageRepairTool:
    """Image repair and validation tool"""
    
    def __init__(self, image_dir: str = "image"):
        self.image_dir = image_dir
        self.backup_dir = os.path.join(image_dir, "backup")
        
    def create_backup(self):
        """Create backup of original images"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            logger.info(f"📁 Created backup directory: {self.backup_dir}")
        
        image_files = glob.glob(os.path.join(self.image_dir, "*.jpg"))
        for img_file in image_files:
            if not os.path.exists(os.path.join(self.backup_dir, os.path.basename(img_file))):
                shutil.copy2(img_file, self.backup_dir)
                logger.info(f"💾 Backed up: {os.path.basename(img_file)}")
    
    def validate_images(self) -> List[str]:
        """Validate all images and return list of valid ones"""
        image_files = glob.glob(os.path.join(self.image_dir, "*.jpg"))
        valid_images = []
        corrupted_images = []
        
        for img_path in sorted(image_files):
            try:
                # Try with OpenCV
                img_cv = cv2.imread(img_path)
                if img_cv is not None:
                    # Try with PIL as double check
                    img_pil = Image.open(img_path)
                    img_pil.verify()
                    valid_images.append(img_path)
                    logger.info(f"✅ Valid image: {os.path.basename(img_path)} - Shape: {img_cv.shape}")
                else:
                    corrupted_images.append(img_path)
                    logger.error(f"❌ OpenCV failed to load: {os.path.basename(img_path)}")
            except Exception as e:
                corrupted_images.append(img_path)
                logger.error(f"❌ Corrupted image: {os.path.basename(img_path)} - Error: {e}")
        
        if corrupted_images:
            logger.warning(f"🚨 Found {len(corrupted_images)} corrupted images")
            for img in corrupted_images:
                logger.warning(f"   - {os.path.basename(img)}")
        
        return valid_images
    
    def repair_jpeg(self, img_path: str) -> bool:
        """Attempt to repair corrupted JPEG"""
        try:
            # Try to open with PIL and re-save
            img = Image.open(img_path)
            img = img.convert('RGB')
            
            # Create repaired filename
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            repaired_path = os.path.join(self.image_dir, f"{base_name}_repaired.jpg")
            
            # Save with high quality
            img.save(repaired_path, 'JPEG', quality=95, optimize=True)
            logger.info(f"🔧 Repaired image saved as: {os.path.basename(repaired_path)}")
            return True
        except Exception as e:
            logger.error(f"🔧 Failed to repair {os.path.basename(img_path)}: {e}")
            return False

def test_stitching_parameters():
    """Test different parameter combinations for better results"""
    from GeometricTransformofImage import ImageStitcher
    
    # Find valid images
    repair_tool = ImageRepairTool()
    valid_images = repair_tool.validate_images()
    
    if len(valid_images) < 2:
        logger.error("❌ Need at least 2 valid images for stitching")
        return
    
    # Test different parameter combinations
    test_configs = [
        # More lenient parameters for better matching
        {'detector': 'SIFT', 'matcher': 'FLANN', 'transform': 'homography', 'blend': 'average'},
        {'detector': 'ORB', 'matcher': 'BF', 'transform': 'homography', 'blend': 'average'},
        {'detector': 'AKAZE', 'matcher': 'BF', 'transform': 'affine', 'blend': 'average'},
        {'detector': 'SIFT', 'matcher': 'BF', 'transform': 'affine', 'blend': 'average'},
    ]
    
    logger.info(f"🧪 Testing {len(test_configs)} parameter combinations with {len(valid_images)} images...")
    
    for i, config in enumerate(test_configs):
        logger.info(f"\n📊 Test {i+1}: {config['detector']}-{config['matcher']}-{config['transform']}-{config['blend']}")
        
        try:
            stitcher = ImageStitcher(
                detector_type=config['detector'],
                matcher_type=config['matcher'],
                transform_type=config['transform'],
                blend_type=config['blend']
            )
            
            result = stitcher.stitch_images(valid_images[:3])  # Use first 3 valid images
            
            if result is not None:
                output_path = f"results/test_{i+1}_{config['detector']}_{config['transform']}.jpg"
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(output_path, result)
                logger.info(f"✅ Success! Result saved to: {output_path}")
                
                # Print statistics
                stitcher.print_statistics()
                
                # Save intermediate results for analysis
                stitcher.save_intermediate_results(f"results/debug_{i+1}")
                
                return  # Stop on first success
            else:
                logger.warning(f"❌ Test {i+1} failed")
                
        except Exception as e:
            logger.error(f"❌ Test {i+1} error: {e}")
    
    logger.error("🚨 All parameter combinations failed. Please check image quality and overlap.")

def analyze_image_overlap():
    """Analyze if images have sufficient overlap"""
    repair_tool = ImageRepairTool()
    valid_images = repair_tool.validate_images()
    
    if len(valid_images) < 2:
        return
    
    logger.info("\n🔍 Analyzing image overlap...")
    
    # Simple overlap analysis using SIFT features
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()
    
    for i in range(len(valid_images)-1):
        img1 = cv2.imread(valid_images[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(valid_images[i+1], cv2.IMREAD_GRAYSCALE)
        
        # Detect features
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        
        if des1 is not None and des2 is not None:
            # Match features
            matches = matcher.knnMatch(des1, des2, k=2)
            
            # Apply NNDR test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            overlap_ratio = len(good_matches) / min(len(kp1), len(kp2))
            
            logger.info(f"📐 Images {i+1} & {i+2}: {len(good_matches)} matches, overlap ratio: {overlap_ratio:.3f}")
            
            if overlap_ratio < 0.1:
                logger.warning(f"⚠️  Low overlap between images {i+1} & {i+2}")
            elif overlap_ratio > 0.3:
                logger.info(f"✅ Good overlap between images {i+1} & {i+2}")

def main():
    """Main function"""
    print("🔧 图像拼接问题修复工具")
    print("=" * 50)
    
    # Create repair tool
    repair_tool = ImageRepairTool()
    
    # Create backup
    repair_tool.create_backup()
    
    # Validate images
    valid_images = repair_tool.validate_images()
    
    # Try to repair corrupted images
    corrupted_files = glob.glob("image/*.jpg")
    for img_path in corrupted_files:
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.info(f"🔧 Attempting to repair: {os.path.basename(img_path)}")
                repair_tool.repair_jpeg(img_path)
        except:
            pass
    
    # Re-validate after repair
    valid_images = repair_tool.validate_images()
    
    if len(valid_images) >= 2:
        # Analyze overlap
        analyze_image_overlap()
        
        # Test different parameter combinations
        test_stitching_parameters()
    else:
        logger.error("❌ 需要至少2张有效图像进行拼接测试")
        logger.info("💡 建议：")
        logger.info("   1. 重新拍摄图像，确保足够的重叠度（30-50%）")
        logger.info("   2. 使用更稳定的相机设置")
        logger.info("   3. 避免相机抖动")

if __name__ == "__main__":
    main() 