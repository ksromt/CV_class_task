"""
Image Blending Module for Image Stitching
Implements various blending strategies for seamless panorama creation
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageBlender:
    """
    Image blender class supporting multiple blending strategies
    """
    
    def __init__(self, blend_type: str = 'average'):
        """
        Initialize image blender
        
        Args:
            blend_type: Type of blending ('average', 'weighted', 'feather')
        """
        self.blend_type = blend_type.lower()
        
    def average_blend(self, warped_images: List[np.ndarray], 
                     masks: List[np.ndarray]) -> np.ndarray:
        """
        Average blending of multiple images
        This implements step 7 of the assignment
        
        Args:
            warped_images: List of warped images
            masks: List of corresponding masks
            
        Returns:
            Blended panorama image
        """
        if not warped_images or not masks:
            logger.error("No images or masks provided for blending")
            return np.array([])
        
        # Initialize result arrays
        canvas_shape = warped_images[0].shape
        result = np.zeros(canvas_shape, dtype=np.float64)
        weight_sum = np.zeros(canvas_shape[:2], dtype=np.float64)
        
        logger.info(f"Blending {len(warped_images)} images of shape {canvas_shape}")
        
        for i, (warped_img, mask) in enumerate(zip(warped_images, masks)):
            # Normalize mask to [0, 1]
            normalized_mask = mask.astype(np.float64) / 255.0
            
            # Add weighted contribution to each color channel
            for c in range(canvas_shape[2]):
                result[:, :, c] += warped_img[:, :, c].astype(np.float64) * normalized_mask
            
            # Accumulate weights
            weight_sum += normalized_mask
            
            logger.debug(f"Image {i+1}: Added {np.sum(normalized_mask > 0)} pixels to blend")
        
        # Avoid division by zero
        weight_sum[weight_sum == 0] = 1
        
        # Normalize by accumulated weights
        for c in range(canvas_shape[2]):
            result[:, :, c] /= weight_sum
        
        # Convert back to uint8
        blended_result = np.clip(result, 0, 255).astype(np.uint8)
        
        logger.info(f"Blending completed. Final image shape: {blended_result.shape}")
        return blended_result
    
    def weighted_blend(self, warped_images: List[np.ndarray], 
                      masks: List[np.ndarray],
                      weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Weighted blending with custom weights per image
        
        Args:
            warped_images: List of warped images
            masks: List of corresponding masks
            weights: Optional weights for each image
            
        Returns:
            Blended panorama image
        """
        if weights is None:
            weights = [1.0] * len(warped_images)
        
        if len(weights) != len(warped_images):
            logger.warning("Weight count mismatch, using equal weights")
            weights = [1.0] * len(warped_images)
        
        # Initialize result arrays
        canvas_shape = warped_images[0].shape
        result = np.zeros(canvas_shape, dtype=np.float64)
        weight_sum = np.zeros(canvas_shape[:2], dtype=np.float64)
        
        for i, (warped_img, mask, weight) in enumerate(zip(warped_images, masks, weights)):
            # Normalize mask and apply weight
            normalized_mask = mask.astype(np.float64) / 255.0 * weight
            
            # Add weighted contribution
            for c in range(canvas_shape[2]):
                result[:, :, c] += warped_img[:, :, c].astype(np.float64) * normalized_mask
            
            # Accumulate weights
            weight_sum += normalized_mask
            
            logger.debug(f"Image {i+1}: Weight={weight:.2f}, pixels={np.sum(normalized_mask > 0)}")
        
        # Normalize
        weight_sum[weight_sum == 0] = 1
        for c in range(canvas_shape[2]):
            result[:, :, c] /= weight_sum
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def feather_blend(self, warped_images: List[np.ndarray], 
                     masks: List[np.ndarray],
                     feather_radius: int = 20) -> np.ndarray:
        """
        Feathered blending for smoother transitions
        
        Args:
            warped_images: List of warped images
            masks: List of corresponding masks
            feather_radius: Radius for feathering effect
            
        Returns:
            Blended panorama image
        """
        # Create feathered masks
        feathered_masks = []
        
        for mask in masks:
            # Apply distance transform for feathering
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Compute distance from edges
            dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            
            # Create feathered weight
            feathered_weight = np.minimum(dist_transform / feather_radius, 1.0)
            feathered_masks.append(feathered_weight)
        
        # Blend using feathered weights
        canvas_shape = warped_images[0].shape
        result = np.zeros(canvas_shape, dtype=np.float64)
        weight_sum = np.zeros(canvas_shape[:2], dtype=np.float64)
        
        for warped_img, feathered_mask in zip(warped_images, feathered_masks):
            # Add weighted contribution
            for c in range(canvas_shape[2]):
                result[:, :, c] += warped_img[:, :, c].astype(np.float64) * feathered_mask
            
            weight_sum += feathered_mask
        
        # Normalize
        weight_sum[weight_sum == 0] = 1
        for c in range(canvas_shape[2]):
            result[:, :, c] /= weight_sum
        
        logger.info(f"Feather blending completed with radius {feather_radius}")
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def blend_images(self, warped_images: List[np.ndarray], 
                    masks: List[np.ndarray],
                    **kwargs) -> np.ndarray:
        """
        Main blending function that dispatches to specific blending method
        
        Args:
            warped_images: List of warped images
            masks: List of corresponding masks
            **kwargs: Additional parameters for specific blending methods
            
        Returns:
            Blended panorama image
        """
        if self.blend_type == 'average':
            return self.average_blend(warped_images, masks)
        elif self.blend_type == 'weighted':
            weights = kwargs.get('weights', None)
            return self.weighted_blend(warped_images, masks, weights)
        elif self.blend_type == 'feather':
            feather_radius = kwargs.get('feather_radius', 20)
            return self.feather_blend(warped_images, masks, feather_radius)
        else:
            logger.warning(f"Unknown blend type '{self.blend_type}', using average")
            return self.average_blend(warped_images, masks)
    
    def analyze_overlap_regions(self, masks: List[np.ndarray]) -> dict:
        """
        Analyze overlap regions between images
        
        Args:
            masks: List of binary masks
            
        Returns:
            Dictionary with overlap analysis
        """
        if not masks:
            return {}
        
        # Count overlaps
        overlap_count = np.zeros(masks[0].shape, dtype=np.int32)
        for mask in masks:
            binary_mask = (mask > 0).astype(np.int32)
            overlap_count += binary_mask
        
        # Analyze overlap statistics
        unique_overlaps, counts = np.unique(overlap_count, return_counts=True)
        
        analysis = {
            'max_overlap': int(np.max(overlap_count)),
            'mean_overlap': float(np.mean(overlap_count[overlap_count > 0])),
            'overlap_distribution': dict(zip(unique_overlaps.tolist(), counts.tolist())),
            'total_pixels': int(np.prod(masks[0].shape)),
            'covered_pixels': int(np.sum(overlap_count > 0))
        }
        
        # Calculate coverage percentage
        analysis['coverage_percentage'] = (analysis['covered_pixels'] / analysis['total_pixels']) * 100
        
        logger.info(f"Overlap analysis: max={analysis['max_overlap']}, coverage={analysis['coverage_percentage']:.1f}%")
        return analysis
    
    def create_seam_mask(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Create optimal seam between two overlapping regions
        
        Args:
            mask1: First image mask
            mask2: Second image mask
            
        Returns:
            Seam mask (0=first image, 255=second image)
        """
        # Find overlap region
        overlap = ((mask1 > 0) & (mask2 > 0)).astype(np.uint8)
        
        if np.sum(overlap) == 0:
            # No overlap, return original masks
            return mask2
        
        # Simple seam - divide overlap region in half
        # For more advanced implementation, could use graph cuts or dynamic programming
        overlap_coords = np.where(overlap > 0)
        if len(overlap_coords[0]) > 0:
            center_y = np.mean(overlap_coords[0])
            
            # Create seam mask
            seam_mask = mask2.copy()
            seam_mask[overlap_coords[0] < center_y, overlap_coords[1][overlap_coords[0] < center_y]] = 0
        else:
            seam_mask = mask2
        
        return seam_mask
    
    def progressive_blend(self, warped_images: List[np.ndarray], 
                         masks: List[np.ndarray]) -> np.ndarray:
        """
        Progressive blending - blend images one by one
        
        Args:
            warped_images: List of warped images
            masks: List of corresponding masks
            
        Returns:
            Blended panorama image
        """
        if not warped_images:
            return np.array([])
        
        # Start with first image
        result = warped_images[0].copy().astype(np.float64)
        result_mask = masks[0].copy()
        
        for i in range(1, len(warped_images)):
            current_img = warped_images[i].astype(np.float64)
            current_mask = masks[i]
            
            # Find overlap regions
            overlap = ((result_mask > 0) & (current_mask > 0)).astype(np.float64) / 255.0
            only_current = ((result_mask == 0) & (current_mask > 0)).astype(np.float64) / 255.0
            
            # Blend in overlap regions (50-50 blend)
            for c in range(result.shape[2]):
                result[:, :, c] = (result[:, :, c] * (1 - overlap * 0.5) + 
                                 current_img[:, :, c] * overlap * 0.5 +
                                 current_img[:, :, c] * only_current)
            
            # Update result mask
            result_mask = np.maximum(result_mask, current_mask)
            
            logger.debug(f"Progressive blend: added image {i+1}")
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def compute_blend_quality(self, blended_image: np.ndarray, 
                            masks: List[np.ndarray]) -> dict:
        """
        Compute quality metrics for blended result
        
        Args:
            blended_image: Final blended image
            masks: Original masks
            
        Returns:
            Dictionary with quality metrics
        """
        if blended_image.size == 0:
            return {}
        
        # Basic image statistics
        gray_result = cv2.cvtColor(blended_image, cv2.COLOR_BGR2GRAY) if len(blended_image.shape) == 3 else blended_image
        
        quality_metrics = {
            'mean_intensity': float(np.mean(gray_result)),
            'std_intensity': float(np.std(gray_result)),
            'contrast': float(np.std(gray_result) / np.mean(gray_result)) if np.mean(gray_result) > 0 else 0,
            'dynamic_range': int(np.max(gray_result) - np.min(gray_result)),
            'filled_pixels': int(np.sum(gray_result > 0)),
            'total_pixels': int(np.prod(gray_result.shape))
        }
        
        # Calculate fill ratio
        quality_metrics['fill_ratio'] = quality_metrics['filled_pixels'] / quality_metrics['total_pixels']
        
        # Analyze overlap coverage
        overlap_analysis = self.analyze_overlap_regions(masks)
        quality_metrics.update(overlap_analysis)
        
        return quality_metrics


def compare_blend_methods(warped_images: List[np.ndarray], 
                         masks: List[np.ndarray]) -> dict:
    """
    Compare different blending methods
    
    Args:
        warped_images: List of warped images
        masks: List of corresponding masks
        
    Returns:
        Dictionary with comparison results
    """
    blend_methods = ['average', 'weighted', 'feather']
    results = {}
    
    for method in blend_methods:
        try:
            blender = ImageBlender(method)
            
            if method == 'weighted':
                # Use decreasing weights for demonstration
                weights = [1.0 / (i + 1) for i in range(len(warped_images))]
                blended = blender.blend_images(warped_images, masks, weights=weights)
            elif method == 'feather':
                blended = blender.blend_images(warped_images, masks, feather_radius=30)
            else:
                blended = blender.blend_images(warped_images, masks)
            
            # Compute quality metrics
            quality = blender.compute_blend_quality(blended, masks)
            
            results[method] = {
                'blended_image': blended,
                'quality_metrics': quality,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Blending method '{method}' failed: {e}")
            results[method] = {
                'blended_image': None,
                'quality_metrics': {},
                'success': False,
                'error': str(e)
            }
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic test data
    np.random.seed(42)
    
    # Create test images with some overlap
    height, width = 400, 600
    test_images = []
    test_masks = []
    
    for i in range(3):
        # Create random image
        img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        
        # Create mask with some overlap
        mask = np.zeros((height, width), dtype=np.uint8)
        start_x = i * 150
        end_x = min(start_x + 300, width)
        mask[:, start_x:end_x] = 255
        
        test_images.append(img)
        test_masks.append(mask)
    
    # Test blending
    blender = ImageBlender('average')
    
    # Average blending
    result = blender.blend_images(test_images, test_masks)
    print(f"Average blend result shape: {result.shape}")
    
    # Analyze overlap
    overlap_analysis = blender.analyze_overlap_regions(test_masks)
    print("Overlap Analysis:")
    for key, value in overlap_analysis.items():
        print(f"  {key}: {value}")
    
    # Compare methods
    comparison = compare_blend_methods(test_images, test_masks)
    print("\nBlending Method Comparison:")
    for method, result in comparison.items():
        if result['success']:
            quality = result['quality_metrics']
            print(f"{method:8}: Fill ratio = {quality.get('fill_ratio', 0):.3f}, "
                  f"Contrast = {quality.get('contrast', 0):.3f}")
        else:
            print(f"{method:8}: FAILED - {result.get('error', 'Unknown error')}")
    
    print("Blending tests completed successfully!") 