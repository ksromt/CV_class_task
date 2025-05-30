"""
Image Warping Module for Image Stitching
Handles canvas size computation and image geometric transformations
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ImageWarper:
    """
    Image warper class for geometric transformations and canvas management
    """
    
    def __init__(self, border_padding: int = 10):
        """
        Initialize image warper
        
        Args:
            border_padding: Padding around the final canvas
        """
        self.border_padding = border_padding
    
    def calculate_canvas_size(self, images: List[np.ndarray], 
                            transformations: List[Optional[np.ndarray]]) -> Tuple[int, int, float, float]:
        """
        Calculate the size of the resulting composite canvas
        This implements step 5 of the assignment
        
        Args:
            images: List of input images
            transformations: List of transformation matrices
            
        Returns:
            Tuple of (canvas_width, canvas_height, min_x, min_y)
        """
        all_corners = []
        
        for i, (img, M) in enumerate(zip(images, transformations)):
            h, w = img.shape[:2]
            
            # Get image corners in homogeneous coordinates
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], 
                             dtype=np.float32).reshape(-1, 1, 2)
            
            if M is not None:
                try:
                    if M.shape[0] == 2:  # Affine transformation (2x3)
                        # Convert to 3x3 homogeneous matrix
                        M_homo = np.vstack([M, [0, 0, 1]])
                        transformed_corners = cv2.perspectiveTransform(corners, M_homo)
                    else:  # Perspective transformation (3x3)
                        transformed_corners = cv2.perspectiveTransform(corners, M)
                    
                    all_corners.extend(transformed_corners.reshape(-1, 2))
                    logger.debug(f"Image {i+1}: Transformed corners computed")
                    
                except Exception as e:
                    logger.warning(f"Failed to transform corners for image {i+1}: {e}")
                    # Fall back to original corners
                    all_corners.extend(corners.reshape(-1, 2))
            else:
                # No transformation (reference image)
                all_corners.extend(corners.reshape(-1, 2))
        
        if not all_corners:
            logger.error("No valid corners found")
            return 800, 600, 0, 0
        
        all_corners = np.array(all_corners)
        
        # Calculate bounding box
        min_x, min_y = np.min(all_corners, axis=0)
        max_x, max_y = np.max(all_corners, axis=0)
        
        # Canvas size with padding
        canvas_width = int(max_x - min_x) + 2 * self.border_padding
        canvas_height = int(max_y - min_y) + 2 * self.border_padding
        
        logger.info(f"Canvas size calculated: {canvas_width} x {canvas_height}")
        logger.info(f"Coordinate range: x[{min_x:.1f}, {max_x:.1f}], y[{min_y:.1f}, {max_y:.1f}]")
        
        return canvas_width, canvas_height, min_x, min_y
    
    def warp_image(self, image: np.ndarray, transformation: Optional[np.ndarray], 
                   canvas_size: Tuple[int, int, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp single image to canvas coordinates
        This implements step 6 of the assignment
        
        Args:
            image: Input image
            transformation: Transformation matrix (None for reference image)
            canvas_size: Canvas dimensions and offset
            
        Returns:
            Tuple of (warped_image, mask)
        """
        canvas_width, canvas_height, min_x, min_y = canvas_size
        
        # Create offset matrix to handle negative coordinates
        offset_matrix = np.array([[1, 0, -min_x + self.border_padding], 
                                [0, 1, -min_y + self.border_padding], 
                                [0, 0, 1]], dtype=np.float32)
        
        if transformation is not None:
            # Combine transformation with offset
            if transformation.shape[0] == 2:  # Affine (2x3)
                transformation_homo = np.vstack([transformation, [0, 0, 1]])
            else:  # Perspective (3x3)
                transformation_homo = transformation
            
            final_transform = offset_matrix @ transformation_homo
        else:
            # Reference image - only apply offset
            final_transform = offset_matrix
        
        try:
            # Warp image
            warped_image = cv2.warpPerspective(
                image, final_transform, 
                (canvas_width, canvas_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Create mask for valid pixels
            mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(
                mask, final_transform,
                (canvas_width, canvas_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            logger.debug(f"Image warped successfully to {canvas_width}x{canvas_height}")
            return warped_image, warped_mask
            
        except Exception as e:
            logger.error(f"Image warping failed: {e}")
            # Return empty image and mask
            warped_image = np.zeros((canvas_height, canvas_width, 3), dtype=image.dtype)
            warped_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
            return warped_image, warped_mask
    
    def warp_all_images(self, images: List[np.ndarray], 
                       transformations: List[Optional[np.ndarray]]) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple]:
        """
        Warp all images to a common canvas
        
        Args:
            images: List of input images
            transformations: List of transformation matrices
            
        Returns:
            Tuple of (warped_images, masks, canvas_info)
        """
        # Calculate canvas size
        canvas_size = self.calculate_canvas_size(images, transformations)
        
        warped_images = []
        masks = []
        
        for i, (img, transform) in enumerate(zip(images, transformations)):
            warped_img, mask = self.warp_image(img, transform, canvas_size)
            warped_images.append(warped_img)
            masks.append(mask)
            
            logger.info(f"Image {i+1} warped - Non-zero pixels: {np.count_nonzero(mask)}")
        
        return warped_images, masks, canvas_size
    
    def get_overlap_regions(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        Compute overlap regions between images
        
        Args:
            masks: List of binary masks
            
        Returns:
            Overlap count matrix
        """
        if not masks:
            return np.array([])
        
        # Initialize overlap counter
        overlap_count = np.zeros(masks[0].shape, dtype=np.int32)
        
        # Count overlaps
        for mask in masks:
            binary_mask = (mask > 0).astype(np.int32)
            overlap_count += binary_mask
        
        logger.info(f"Overlap analysis: max overlap = {np.max(overlap_count)} images")
        return overlap_count
    
    def visualize_canvas_layout(self, images: List[np.ndarray], 
                              transformations: List[Optional[np.ndarray]],
                              save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize the layout of images on canvas
        
        Args:
            images: List of input images
            transformations: List of transformation matrices
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        canvas_size = self.calculate_canvas_size(images, transformations)
        canvas_width, canvas_height, min_x, min_y = canvas_size
        
        # Create visualization canvas
        vis_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Color palette for different images
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0)]
        
        for i, (img, transform) in enumerate(zip(images, transformations)):
            h, w = img.shape[:2]
            color = colors[i % len(colors)]
            
            # Get image corners
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], 
                             dtype=np.float32).reshape(-1, 1, 2)
            
            # Transform corners
            offset_matrix = np.array([[1, 0, -min_x + self.border_padding], 
                                    [0, 1, -min_y + self.border_padding], 
                                    [0, 0, 1]], dtype=np.float32)
            
            if transform is not None:
                if transform.shape[0] == 2:
                    transform_homo = np.vstack([transform, [0, 0, 1]])
                else:
                    transform_homo = transform
                final_transform = offset_matrix @ transform_homo
            else:
                final_transform = offset_matrix
            
            try:
                transformed_corners = cv2.perspectiveTransform(corners, final_transform)
                
                # Draw image boundary
                pts = transformed_corners.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(vis_canvas, [pts], True, color, 2)
                
                # Add image label
                center = np.mean(transformed_corners.reshape(-1, 2), axis=0).astype(int)
                cv2.putText(vis_canvas, f'IMG{i+1}', tuple(center), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                          
            except Exception as e:
                logger.warning(f"Failed to visualize image {i+1}: {e}")
        
        if save_path:
            cv2.imwrite(save_path, vis_canvas)
            logger.info(f"Canvas layout visualization saved to {save_path}")
        
        return vis_canvas
    
    def compute_transformation_quality(self, src_pts: np.ndarray, dst_pts: np.ndarray, 
                                     transformation: np.ndarray, transform_type: str) -> Dict[str, float]:
        """
        Compute quality metrics for transformation
        
        Args:
            src_pts: Source points
            dst_pts: Destination points
            transformation: Transformation matrix
            transform_type: Type of transformation
            
        Returns:
            Dictionary with quality metrics
        """
        if src_pts is None or dst_pts is None or transformation is None:
            return {}
        
        try:
            # Apply transformation to source points
            if transform_type.lower() in ['homography', 'perspective']:
                projected = cv2.perspectiveTransform(src_pts, transformation)
            else:
                if transformation.shape[0] == 2:
                    transform_homo = np.vstack([transformation, [0, 0, 1]])
                else:
                    transform_homo = transformation
                projected = cv2.perspectiveTransform(src_pts, transform_homo)
            
            # Compute various error metrics
            errors = np.linalg.norm(projected - dst_pts, axis=2).flatten()
            
            quality_metrics = {
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'std_error': np.std(errors),
                'max_error': np.max(errors),
                'rmse': np.sqrt(np.mean(errors**2)),
                'num_points': len(errors)
            }
            
            # Add inlier ratio (assuming errors < 3 pixels are inliers)
            inlier_threshold = 3.0
            inliers = errors < inlier_threshold
            quality_metrics['inlier_ratio'] = np.sum(inliers) / len(errors)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality computation failed: {e}")
            return {}


# Utility functions
def create_grid_points(width: int, height: int, spacing: int = 50) -> np.ndarray:
    """
    Create a grid of points for transformation testing
    
    Args:
        width: Image width
        height: Image height
        spacing: Grid spacing
        
    Returns:
        Array of grid points
    """
    x_coords = np.arange(0, width, spacing)
    y_coords = np.arange(0, height, spacing)
    
    grid_points = []
    for y in y_coords:
        for x in x_coords:
            grid_points.append([x, y])
    
    return np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)


def apply_transformation_to_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """
    Apply transformation to a set of points
    
    Args:
        points: Input points (Nx1x2)
        transformation: Transformation matrix
        
    Returns:
        Transformed points
    """
    try:
        if transformation.shape[0] == 2:  # Affine
            transformation_homo = np.vstack([transformation, [0, 0, 1]])
        else:  # Perspective
            transformation_homo = transformation
        
        transformed = cv2.perspectiveTransform(points, transformation_homo)
        return transformed
        
    except Exception as e:
        logger.error(f"Point transformation failed: {e}")
        return points


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic test data
    np.random.seed(42)
    
    # Create dummy images
    test_images = []
    for i in range(3):
        img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        test_images.append(img)
    
    # Create dummy transformations
    transformations = [
        None,  # Reference image
        np.array([[1.0, 0.0, 50], [0.0, 1.0, 20]], dtype=np.float32),  # Translation
        np.array([[0.9, -0.1, 100], [0.1, 0.9, 40]], dtype=np.float32)  # Similarity
    ]
    
    # Test image warper
    warper = ImageWarper()
    
    # Calculate canvas size
    canvas_size = warper.calculate_canvas_size(test_images, transformations)
    print(f"Canvas size: {canvas_size[0]} x {canvas_size[1]}")
    print(f"Offset: ({canvas_size[2]:.1f}, {canvas_size[3]:.1f})")
    
    # Warp all images
    warped_images, masks, canvas_info = warper.warp_all_images(test_images, transformations)
    
    print(f"Warped {len(warped_images)} images successfully")
    
    # Analyze overlaps
    overlap_count = warper.get_overlap_regions(masks)
    print(f"Maximum overlap: {np.max(overlap_count)} images")
    
    # Create visualization
    vis = warper.visualize_canvas_layout(test_images, transformations, 'results/canvas_layout.jpg')
    print("Canvas layout visualization created") 