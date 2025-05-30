"""
Transform Estimation Module for Image Stitching
Implements translation, similarity, affine, and perspective transformations
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TransformEstimator:
    """
    Transform estimator class supporting multiple transformation types
    """
    
    def __init__(self, ransac_threshold: float = 10.0, confidence: float = 0.995):
        """
        Initialize transform estimator
        
        Args:
            ransac_threshold: RANSAC inlier threshold (increased for better robustness)
            confidence: RANSAC confidence level
        """
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TransformEstimator initialized with threshold={ransac_threshold}")
    
    def estimate_translation(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate translation transformation (2 DOF)
        
        Args:
            src_pts: Source points (Nx1x2)
            dst_pts: Destination points (Nx1x2)
            
        Returns:
            2x3 transformation matrix or None
        """
        if src_pts is None or dst_pts is None or len(src_pts) < 1:
            return None
        
        try:
            # Calculate mean displacement
            displacement = np.mean(dst_pts - src_pts, axis=0)
            
            # Create translation matrix
            M = np.array([[1, 0, displacement[0, 0]], 
                         [0, 1, displacement[0, 1]]], dtype=np.float32)
            
            logger.info(f"Translation estimated: dx={displacement[0, 0]:.2f}, dy={displacement[0, 1]:.2f}")
            return M
            
        except Exception as e:
            logger.error(f"Translation estimation failed: {e}")
            return None
    
    def estimate_similarity(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate similarity transformation (4 DOF: translation, rotation, uniform scale)
        
        Args:
            src_pts: Source points (Nx1x2)
            dst_pts: Destination points (Nx1x2)
            
        Returns:
            2x3 transformation matrix or None
        """
        if src_pts is None or dst_pts is None or len(src_pts) < 2:
            return None
        
        try:
            # Use OpenCV's estimateAffinePartial2D for similarity transform
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                confidence=self.confidence
            )
            
            if M is not None:
                logger.info(f"Similarity transform estimated with {np.sum(inliers)} inliers")
                return M
            else:
                logger.warning("Similarity estimation failed")
                return None
                
        except Exception as e:
            logger.error(f"Similarity estimation failed: {e}")
            return None
    
    def estimate_affine(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate affine transformation (6 DOF)
        
        Args:
            src_pts: Source points (Nx1x2)
            dst_pts: Destination points (Nx1x2)
            
        Returns:
            2x3 transformation matrix or None
        """
        if src_pts is None or dst_pts is None or len(src_pts) < 3:
            return None
        
        try:
            # Use OpenCV's estimateAffine2D
            M, inliers = cv2.estimateAffine2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                confidence=self.confidence
            )
            
            if M is not None:
                logger.info(f"Affine transform estimated with {np.sum(inliers)} inliers")
                return M
            else:
                logger.warning("Affine estimation failed")
                return None
                
        except Exception as e:
            logger.error(f"Affine estimation failed: {e}")
            return None
    
    def estimate_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate homography transformation (8 DOF)
        
        Args:
            src_pts: Source points (Nx2)
            dst_pts: Destination points (Nx2)
            
        Returns:
            3x3 homography matrix or None if estimation fails
        """
        if len(src_pts) < 8:  # Reduced minimum points
            self.logger.warning(f"Insufficient points for homography: {len(src_pts)} < 8")
            return None
        
        try:
            # Estimate homography with more lenient parameters
            H, mask = cv2.findHomography(
                src_pts, dst_pts, 
                cv2.RANSAC, 
                self.ransac_threshold,
                confidence=self.confidence,
                maxIters=5000  # Increased iterations
            )
            
            if H is not None:
                inliers = np.sum(mask)
                total_points = len(src_pts)
                inlier_ratio = inliers / total_points
                
                self.logger.info(f"Homography estimated with {inliers} inliers")
                self.logger.info(f"Inlier ratio: {inlier_ratio:.3f}")
                
                # More lenient inlier ratio check
                if inlier_ratio < 0.05:  # Reduced from 0.1 to 0.05
                    self.logger.warning(f"Low inlier ratio: {inlier_ratio:.3f}")
                
                return H
            else:
                self.logger.error("Homography estimation failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in homography estimation: {e}")
            return None
    
    def estimate_transformation(self, src_pts: np.ndarray, dst_pts: np.ndarray, 
                              transform_type: str) -> Optional[np.ndarray]:
        """
        Estimate transformation of specified type
        
        Args:
            src_pts: Source points
            dst_pts: Destination points
            transform_type: Type of transformation
            
        Returns:
            Transformation matrix or None
        """
        transform_type = transform_type.lower()
        
        if transform_type == 'translation':
            return self.estimate_translation(src_pts, dst_pts)
        elif transform_type == 'similarity':
            return self.estimate_similarity(src_pts, dst_pts)
        elif transform_type == 'affine':
            return self.estimate_affine(src_pts, dst_pts)
        elif transform_type == 'homography' or transform_type == 'perspective':
            return self.estimate_homography(src_pts, dst_pts)
        else:
            raise ValueError(f"Unsupported transformation type: {transform_type}")
    
    def compute_reprojection_error(self, src_pts: np.ndarray, dst_pts: np.ndarray, 
                                 M: np.ndarray, transform_type: str) -> float:
        """
        Compute reprojection error for given transformation
        
        Args:
            src_pts: Source points
            dst_pts: Destination points
            M: Transformation matrix
            transform_type: Type of transformation
            
        Returns:
            Mean reprojection error
        """
        if src_pts is None or dst_pts is None or M is None:
            return float('inf')
        
        try:
            if transform_type.lower() == 'homography' or transform_type.lower() == 'perspective':
                # For homography, use perspectiveTransform
                projected = cv2.perspectiveTransform(src_pts, M)
            else:
                # For affine transforms, need to convert to homogeneous coordinates
                if M.shape[0] == 2:  # 2x3 matrix
                    M_homo = np.vstack([M, [0, 0, 1]])
                else:  # already 3x3
                    M_homo = M
                projected = cv2.perspectiveTransform(src_pts, M_homo)
            
            # Calculate mean squared error
            errors = np.linalg.norm(projected - dst_pts, axis=2)
            mean_error = np.mean(errors)
            
            return mean_error
            
        except Exception as e:
            logger.error(f"Error computation failed: {e}")
            return float('inf')
    
    def compare_transformations(self, src_pts: np.ndarray, dst_pts: np.ndarray, 
                              transform_types: list = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare different transformation types on the same point set
        
        Args:
            src_pts: Source points
            dst_pts: Destination points
            transform_types: List of transform types to compare
            
        Returns:
            Dictionary with comparison results
        """
        if transform_types is None:
            transform_types = ['translation', 'similarity', 'affine', 'homography']
        
        results = {}
        
        for transform_type in transform_types:
            try:
                # Estimate transformation
                M = self.estimate_transformation(src_pts, dst_pts, transform_type)
                
                if M is not None:
                    # Compute reprojection error
                    error = self.compute_reprojection_error(src_pts, dst_pts, M, transform_type)
                    
                    # Get degrees of freedom
                    dof_map = {
                        'translation': 2,
                        'similarity': 4,
                        'affine': 6,
                        'homography': 8,
                        'perspective': 8
                    }
                    
                    results[transform_type] = {
                        'matrix': M,
                        'error': error,
                        'dof': dof_map.get(transform_type, 0),
                        'success': True
                    }
                else:
                    results[transform_type] = {
                        'matrix': None,
                        'error': float('inf'),
                        'dof': 0,
                        'success': False
                    }
                    
            except Exception as e:
                logger.error(f"Failed to estimate {transform_type}: {e}")
                results[transform_type] = {
                    'matrix': None,
                    'error': float('inf'),
                    'dof': 0,
                    'success': False,
                    'error_msg': str(e)
                }
        
        return results
    
    def analyze_transformation_matrix(self, M: np.ndarray, transform_type: str) -> Dict[str, float]:
        """
        Analyze transformation matrix to extract geometric parameters
        
        Args:
            M: Transformation matrix
            transform_type: Type of transformation
            
        Returns:
            Dictionary with transformation parameters
        """
        analysis = {}
        
        try:
            if transform_type.lower() == 'translation':
                analysis['tx'] = M[0, 2]
                analysis['ty'] = M[1, 2]
                
            elif transform_type.lower() == 'similarity':
                # Extract rotation, scale, and translation
                a, b = M[0, 0], M[0, 1]
                scale = np.sqrt(a*a + b*b)
                rotation = np.arctan2(b, a) * 180 / np.pi
                
                analysis['scale'] = scale
                analysis['rotation_deg'] = rotation
                analysis['tx'] = M[0, 2]
                analysis['ty'] = M[1, 2]
                
            elif transform_type.lower() == 'affine':
                # Extract scale and shear
                a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
                
                scale_x = np.sqrt(a*a + c*c)
                scale_y = np.sqrt(b*b + d*d)
                shear = (a*b + c*d) / (scale_x * scale_y)
                
                analysis['scale_x'] = scale_x
                analysis['scale_y'] = scale_y
                analysis['shear'] = shear
                analysis['tx'] = M[0, 2]
                analysis['ty'] = M[1, 2]
                
            elif transform_type.lower() in ['homography', 'perspective']:
                # Extract perspective parameters
                analysis['h00'] = M[0, 0]
                analysis['h01'] = M[0, 1]
                analysis['h02'] = M[0, 2]
                analysis['h10'] = M[1, 0]
                analysis['h11'] = M[1, 1]
                analysis['h12'] = M[1, 2]
                analysis['h20'] = M[2, 0]
                analysis['h21'] = M[2, 1]
                analysis['h22'] = M[2, 2]
                
                # Normalize by h22
                if M[2, 2] != 0:
                    for key in analysis:
                        analysis[key] /= M[2, 2]
                        
        except Exception as e:
            logger.error(f"Matrix analysis failed: {e}")
            
        return analysis


def create_synthetic_transformation(transform_type: str, **params) -> np.ndarray:
    """
    Create synthetic transformation matrix for testing
    
    Args:
        transform_type: Type of transformation
        **params: Transformation parameters
        
    Returns:
        Transformation matrix
    """
    if transform_type.lower() == 'translation':
        tx = params.get('tx', 10)
        ty = params.get('ty', 5)
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        
    elif transform_type.lower() == 'similarity':
        scale = params.get('scale', 1.1)
        angle = params.get('angle', 5) * np.pi / 180  # degrees to radians
        tx = params.get('tx', 10)
        ty = params.get('ty', 5)
        
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        M = np.array([[scale*cos_a, -scale*sin_a, tx],
                     [scale*sin_a, scale*cos_a, ty]], dtype=np.float32)
        
    elif transform_type.lower() == 'affine':
        M = np.array([[1.1, 0.1, 10],
                     [0.05, 1.05, 5]], dtype=np.float32)
        
    elif transform_type.lower() in ['homography', 'perspective']:
        M = np.array([[1.1, 0.1, 10],
                     [0.05, 1.05, 5],
                     [0.0001, 0.0002, 1]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported transformation type: {transform_type}")
    
    return M


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic test data
    np.random.seed(42)
    
    # Generate random source points
    n_points = 50
    src_points = np.random.rand(n_points, 1, 2) * 500
    
    # Create ground truth transformation
    gt_transform = create_synthetic_transformation('homography', angle=10, scale=1.2)
    
    # Apply transformation to get destination points
    if gt_transform.shape[0] == 3:
        dst_points = cv2.perspectiveTransform(src_points, gt_transform)
    else:
        gt_homo = np.vstack([gt_transform, [0, 0, 1]])
        dst_points = cv2.perspectiveTransform(src_points, gt_homo)
    
    # Add some noise
    noise = np.random.normal(0, 2, dst_points.shape)
    dst_points_noisy = dst_points + noise
    
    # Test transform estimator
    estimator = TransformEstimator()
    
    # Compare different transformation types
    results = estimator.compare_transformations(src_points, dst_points_noisy)
    
    print("Transformation Comparison Results:")
    print("=" * 60)
    for transform_type, result in results.items():
        if result['success']:
            print(f"{transform_type:12}: Error = {result['error']:8.2f}, DOF = {result['dof']}")
            
            # Analyze matrix
            analysis = estimator.analyze_transformation_matrix(result['matrix'], transform_type)
            if analysis:
                print(f"              Parameters: {analysis}")
        else:
            print(f"{transform_type:12}: FAILED")
        print()
    
    # Find best transformation
    successful_results = {k: v for k, v in results.items() if v['success']}
    if successful_results:
        best_transform = min(successful_results.keys(), 
                           key=lambda k: successful_results[k]['error'])
        print(f"Best transformation: {best_transform} (error: {successful_results[best_transform]['error']:.2f})") 