"""
Main Image Stitching Program
Integrates all modules to create panoramic images from multiple overlapping photos
"""

import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Import custom modules
from src.feature_detector import FeatureDetector
from src.feature_matcher import FeatureMatcher, create_track_indices
from src.transform_estimator import TransformEstimator
from src.image_warper import ImageWarper
from src.image_blender import ImageBlender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stitching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImageStitcher:
    """
    Complete image stitching pipeline
    """
    
    def __init__(self, 
                 detector_type: str = 'SIFT',
                 matcher_type: str = 'BF',
                 transform_type: str = 'homography',
                 blend_type: str = 'average'):
        """
        Initialize image stitcher with configuration
        
        Args:
            detector_type: Feature detector type ('SIFT', 'ORB', 'AKAZE')
            matcher_type: Feature matcher type ('BF', 'FLANN')
            transform_type: Transform type ('translation', 'similarity', 'affine', 'homography')
            blend_type: Blending type ('average', 'weighted', 'feather')
        """
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        self.transform_type = transform_type
        self.blend_type = blend_type
        
        # Initialize components
        self.detector = FeatureDetector(detector_type)
        self.matcher = FeatureMatcher(matcher_type)
        self.estimator = TransformEstimator()
        self.warper = ImageWarper()
        self.blender = ImageBlender(blend_type)
        
        # Results storage
        self.images = []
        self.features_data = []
        self.transformations = []
        self.warped_images = []
        self.masks = []
        self.result = None
        self.stats = {}
        
        logger.info(f"ImageStitcher initialized: {detector_type}-{matcher_type}-{transform_type}-{blend_type}")
    
    def load_images(self, image_paths: List[str]) -> bool:
        """
        Load multiple overlapping images
        This implements step 1 of the assignment
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Success flag
        """
        self.images = []
        
        for i, path in enumerate(image_paths):
            if not os.path.exists(path):
                logger.error(f"Image not found: {path}")
                continue
                
            img = cv2.imread(path)
            if img is not None:
                # Convert BGR to RGB for consistency
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images.append(img_rgb)
                logger.info(f"Loaded image {i+1}: {path} - Shape: {img_rgb.shape}")
            else:
                logger.error(f"Failed to load image: {path}")
        
        if len(self.images) < 2:
            logger.error("Need at least 2 images for stitching")
            return False
        
        logger.info(f"Successfully loaded {len(self.images)} images")
        return True
    
    def detect_features(self) -> bool:
        """
        Detect feature points from all images
        This implements step 2 of the assignment
        
        Returns:
            Success flag
        """
        if not self.images:
            logger.error("No images loaded")
            return False
        
        logger.info("Starting feature detection...")
        start_time = time.time()
        
        # Convert to grayscale for feature detection
        gray_images = []
        for img in self.images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_images.append(gray)
        
        # Detect features
        self.features_data = self.detector.detect_multiple_images(gray_images)
        
        detection_time = time.time() - start_time
        total_features = sum(len(features[0]) for features in self.features_data)
        
        logger.info(f"Feature detection completed in {detection_time:.2f}s")
        logger.info(f"Total features detected: {total_features}")
        
        # Store statistics
        self.stats['feature_detection'] = {
            'time': detection_time,
            'total_features': total_features,
            'features_per_image': [len(features[0]) for features in self.features_data]
        }
        
        return len(self.features_data) > 0
    
    def create_feature_tracks(self) -> bool:
        """
        Create unique track indices for features
        This implements step 3 of the assignment
        
        Returns:
            Success flag
        """
        if not self.features_data:
            logger.error("No features detected")
            return False
        
        logger.info("Creating feature tracks...")
        start_time = time.time()
        
        # Create track indices
        self.track_indices = create_track_indices(self.features_data, self.matcher)
        
        tracking_time = time.time() - start_time
        total_tracks = max(max(tracks) for tracks in self.track_indices.values()) + 1
        
        logger.info(f"Feature tracking completed in {tracking_time:.2f}s")
        logger.info(f"Total unique tracks created: {total_tracks}")
        
        # Store statistics
        self.stats['feature_tracking'] = {
            'time': tracking_time,
            'total_tracks': total_tracks,
            'tracks_per_image': [len(tracks) for tracks in self.track_indices.values()]
        }
        
        return True
    
    def estimate_transformations(self) -> bool:
        """
        Estimate transformations for each image
        This implements step 4 of the assignment
        
        Returns:
            Success flag
        """
        if not self.features_data:
            logger.error("No features available for transformation estimation")
            return False
        
        logger.info(f"Estimating {self.transform_type} transformations...")
        start_time = time.time()
        
        self.transformations = [None]  # Reference image (first image)
        transform_stats = []
        
        # Match and estimate transformation for each image against reference
        reference_features = self.features_data[0]
        
        for i in range(1, len(self.images)):
            # Match features with reference image
            matches = self.matcher.match_features(reference_features, self.features_data[i])
            
            if len(matches) >= 4:  # Minimum points needed
                # Get matched point coordinates
                dst_pts, src_pts = self.matcher.get_matched_points(
                    reference_features[0], self.features_data[i][0], matches
                )
                
                logger.debug(f"Image {i+1}: Found {len(matches)} matches")
                logger.debug(f"Source points shape: {src_pts.shape if src_pts is not None else 'None'}")
                logger.debug(f"Destination points shape: {dst_pts.shape if dst_pts is not None else 'None'}")
                
                # Estimate transformation that maps image i to reference image coordinates
                M = self.estimator.estimate_transformation(src_pts, dst_pts, self.transform_type)
                
                # Validate transformation quality
                if M is not None and self._validate_transformation(M, self.transform_type, src_pts, dst_pts):
                    self.transformations.append(M)
                    
                    # Compute transformation quality
                    error = self.estimator.compute_reprojection_error(
                        src_pts, dst_pts, M, self.transform_type
                    )
                    
                    transform_stats.append({
                        'image_index': i,
                        'matches': len(matches),
                        'reprojection_error': error,
                        'success': True
                    })
                    
                    logger.info(f"Image {i+1}: Transform estimated with {len(matches)} matches, error={error:.2f}")
                    
                    # Log transformation details for debugging
                    if M.shape[0] == 2:  # Affine
                        tx, ty = M[0, 2], M[1, 2]
                        logger.debug(f"  Affine transform: tx={tx:.1f}, ty={ty:.1f}")
                    else:  # Homography
                        logger.debug(f"  Homography matrix:\n{M}")
                        
                else:
                    if M is None:
                        logger.warning(f"Failed to estimate transformation for image {i+1}")
                    else:
                        logger.warning(f"Transformation for image {i+1} failed validation")
                    self.transformations.append(None)
                    transform_stats.append({
                        'image_index': i,
                        'matches': len(matches),
                        'success': False
                    })
            else:
                logger.warning(f"Insufficient matches ({len(matches)}) for image {i+1}")
                self.transformations.append(None)
                transform_stats.append({
                    'image_index': i,
                    'matches': len(matches),
                    'success': False
                })
        
        estimation_time = time.time() - start_time
        successful_transforms = sum(1 for stat in transform_stats if stat['success'])
        
        logger.info(f"Transformation estimation completed in {estimation_time:.2f}s")
        logger.info(f"Successful transformations: {successful_transforms}/{len(transform_stats)}")
        
        # Store statistics
        self.stats['transformation_estimation'] = {
            'time': estimation_time,
            'successful_transforms': successful_transforms,
            'total_attempted': len(transform_stats),
            'details': transform_stats
        }
        
        return successful_transforms > 0
    
    def warp_images(self) -> bool:
        """
        Warp images to canvas and compute canvas size
        This implements steps 5 & 6 of the assignment
        
        Returns:
            Success flag
        """
        if not self.transformations:
            logger.error("No transformations available")
            return False
        
        logger.info("Warping images to canvas...")
        start_time = time.time()
        
        # Warp all images
        self.warped_images, self.masks, self.canvas_info = self.warper.warp_all_images(
            self.images, self.transformations
        )
        
        warping_time = time.time() - start_time
        canvas_width, canvas_height = self.canvas_info[0], self.canvas_info[1]
        
        logger.info(f"Image warping completed in {warping_time:.2f}s")
        logger.info(f"Canvas size: {canvas_width} x {canvas_height}")
        
        # Analyze overlaps
        overlap_analysis = self.warper.get_overlap_regions(self.masks)
        
        # Store statistics
        self.stats['image_warping'] = {
            'time': warping_time,
            'canvas_size': (canvas_width, canvas_height),
            'max_overlap': np.max(overlap_analysis) if overlap_analysis.size > 0 else 0
        }
        
        return len(self.warped_images) > 0
    
    def blend_images(self) -> bool:
        """
        Blend warped images into final panorama
        This implements step 7 of the assignment
        
        Returns:
            Success flag
        """
        if not self.warped_images or not self.masks:
            logger.error("No warped images available for blending")
            return False
        
        logger.info(f"Blending images using {self.blend_type} method...")
        start_time = time.time()
        
        # Blend images
        self.result = self.blender.blend_images(self.warped_images, self.masks)
        
        blending_time = time.time() - start_time
        
        if self.result is not None and self.result.size > 0:
            logger.info(f"Image blending completed in {blending_time:.2f}s")
            logger.info(f"Final panorama shape: {self.result.shape}")
            
            # Compute blend quality
            quality_metrics = self.blender.compute_blend_quality(self.result, self.masks)
            
            # Store statistics
            self.stats['image_blending'] = {
                'time': blending_time,
                'quality_metrics': quality_metrics
            }
            
            return True
        else:
            logger.error("Blending failed")
            return False
    
    def stitch_images(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """
        Complete image stitching pipeline
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Stitched panorama image or None if failed
        """
        logger.info("=" * 60)
        logger.info("STARTING IMAGE STITCHING PIPELINE")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        # Step 1: Load images
        if not self.load_images(image_paths):
            return None
        
        # Step 2: Detect features
        if not self.detect_features():
            return None
        
        # Step 3: Create feature tracks
        if not self.create_feature_tracks():
            return None
        
        # Step 4: Estimate transformations
        if not self.estimate_transformations():
            return None
        
        # Step 5 & 6: Warp images
        if not self.warp_images():
            return None
        
        # Step 7: Blend images
        if not self.blend_images():
            return None
        
        total_time = time.time() - total_start_time
        self.stats['total_time'] = total_time
        
        logger.info("=" * 60)
        logger.info(f"STITCHING COMPLETED SUCCESSFULLY in {total_time:.2f}s")
        logger.info("=" * 60)
        
        return self.result
    
    def save_result(self, output_path: str) -> bool:
        """
        Save stitching result
        
        Args:
            output_path: Output file path
            
        Returns:
            Success flag
        """
        if self.result is None:
            logger.error("No result to save")
            return False
        
        try:
            # Convert RGB back to BGR for OpenCV
            result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(output_path, result_bgr)
            
            if success:
                logger.info(f"Result saved to: {output_path}")
                return True
            else:
                logger.error(f"Failed to save result to: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return False
    
    def save_intermediate_results(self, output_dir: str) -> bool:
        """
        Save intermediate results for analysis
        
        Args:
            output_dir: Output directory
            
        Returns:
            Success flag
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save feature visualizations
            if self.features_data and len(self.images) >= 2:
                gray1 = cv2.cvtColor(self.images[0], cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(self.images[1], cv2.COLOR_RGB2GRAY)
                
                matches = self.matcher.match_features(self.features_data[0], self.features_data[1])
                if matches:
                    match_vis = self.matcher.visualize_matches(
                        gray1, gray2,
                        self.features_data[0][0], self.features_data[1][0],
                        matches[:30],
                        save_path=os.path.join(output_dir, 'feature_matches.jpg')
                    )
            
            # Save warped images
            if self.warped_images:
                for i, warped_img in enumerate(self.warped_images):
                    warped_bgr = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir, f'warped_image_{i+1}.jpg'), warped_bgr)
            
            # Save canvas layout visualization
            if self.images and self.transformations:
                layout_vis = self.warper.visualize_canvas_layout(
                    self.images, self.transformations,
                    save_path=os.path.join(output_dir, 'canvas_layout.jpg')
                )
            
            logger.info(f"Intermediate results saved to: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
            return False
    
    def print_statistics(self) -> None:
        """
        Print detailed statistics about the stitching process
        """
        print("\n" + "=" * 60)
        print("IMAGE STITCHING STATISTICS")
        print("=" * 60)
        
        for stage, stats in self.stats.items():
            print(f"\n{stage.upper().replace('_', ' ')}:")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if key != 'details':
                        print(f"  {key}: {value}")
            else:
                print(f"  {stats}")
        
        print("\n" + "=" * 60)
    
    def compare_methods(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Compare different detector/transform combinations
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Comparison results
        """
        methods_to_test = [
            ('SIFT', 'homography'),
            ('ORB', 'homography'),
            ('AKAZE', 'homography'),
            ('SIFT', 'affine'),
            ('SIFT', 'similarity')
        ]
        
        results = {}
        
        for detector_type, transform_type in methods_to_test:
            logger.info(f"\nTesting {detector_type} + {transform_type}")
            
            # Create new stitcher with this configuration
            stitcher = ImageStitcher(
                detector_type=detector_type,
                transform_type=transform_type
            )
            
            # Run stitching
            result = stitcher.stitch_images(image_paths)
            
            # Store results
            method_key = f"{detector_type}_{transform_type}"
            results[method_key] = {
                'success': result is not None,
                'result_image': result,
                'stats': stitcher.stats.copy() if hasattr(stitcher, 'stats') else {}
            }
        
        return results

    def _validate_transformation(self, M: np.ndarray, transform_type: str, 
                               src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        """
        Validate transformation quality
        
        Args:
            M: Transformation matrix
            transform_type: Type of transformation
            src_pts: Source points
            dst_pts: Destination points
            
        Returns:
            True if transformation is valid
        """
        try:
            # Check for valid matrix
            if M is None or np.any(np.isnan(M)) or np.any(np.isinf(M)):
                logger.warning("Invalid transformation matrix (NaN or Inf)")
                return False
            
            # Compute reprojection error
            error = self.estimator.compute_reprojection_error(src_pts, dst_pts, M, transform_type)
            
            # More lenient error threshold for challenging cases
            max_allowed_error = 100.0  # Increased from 10.0 to 100.0 pixels
            if error > max_allowed_error:
                logger.warning(f"High reprojection error: {error:.2f} > {max_allowed_error}")
                return False
            
            # Check transformation matrix properties
            if transform_type.lower() in ['homography', 'perspective']:
                # Check determinant to avoid degenerate transformations
                det = np.linalg.det(M[:2, :2])
                if abs(det) < 0.01:  # More lenient threshold
                    logger.warning(f"Near-singular transformation (det={det:.4f})")
                    return False
                    
                # More lenient scaling check
                scale = np.sqrt(abs(det))
                if scale < 0.1 or scale > 10.0:  # Relaxed from 0.3-3.0 to 0.1-10.0
                    logger.warning(f"Unreasonable scale factor: {scale:.2f}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Transformation validation failed: {e}")
            return False


def main():
    """
    Main function to run image stitching
    """
    # Image paths (modify these to point to your images)
    image_paths = [
        'image/1.jpg',
        'image/2.jpg', 
        'image/3.jpg',
        'image/4.jpg'
    ]
    
    # Check if images exist
    existing_paths = [path for path in image_paths if os.path.exists(path)]
    
    if len(existing_paths) < 2:
        logger.error("Need at least 2 images in the 'image' folder")
        logger.info("Available images:")
        for file in os.listdir('image'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                logger.info(f"  - {file}")
        return
    
    # Create image stitcher
    stitcher = ImageStitcher(
        detector_type='SIFT',      # Options: 'SIFT', 'ORB', 'AKAZE'
        matcher_type='BF',         # Options: 'BF', 'FLANN'
        transform_type='homography', # Options: 'translation', 'similarity', 'affine', 'homography'
        blend_type='average'       # Options: 'average', 'weighted', 'feather'
    )
    
    # Run stitching
    result = stitcher.stitch_images(existing_paths)
    
    if result is not None:
        # Save results
        os.makedirs('results', exist_ok=True)
        stitcher.save_result('results/panorama_result.jpg')
        stitcher.save_intermediate_results('results')
        
        # Print statistics
        stitcher.print_statistics()
        
        print(f"\n✅ 图像拼接完成！")
        print(f"📁 结果保存在: results/panorama_result.jpg")
        print(f"📊 中间结果保存在: results/ 文件夹")
        
    else:
        print("❌ 图像拼接失败，请检查输入图像和参数设置")


if __name__ == "__main__":
    main() 