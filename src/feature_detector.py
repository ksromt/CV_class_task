"""
Feature Detection Module for Image Stitching
Supports SIFT, ORB, AKAZE feature detectors
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureDetector:
    """
    Feature detector class supporting multiple algorithms
    """
    
    def __init__(self, detector_type: str = 'SIFT', **kwargs):
        """
        Initialize feature detector
        
        Args:
            detector_type: Type of detector ('SIFT', 'ORB', 'AKAZE')
            **kwargs: Additional parameters for specific detectors
        """
        self.detector_type = detector_type.upper()
        self.detector = self._create_detector(**kwargs)
        
    def _create_detector(self, **kwargs):
        """
        Create specific detector instance
        
        Returns:
            OpenCV detector object
        """
        if self.detector_type == 'SIFT':
            nfeatures = kwargs.get('nfeatures', 0)
            nOctaveLayers = kwargs.get('nOctaveLayers', 3)
            contrastThreshold = kwargs.get('contrastThreshold', 0.04)
            edgeThreshold = kwargs.get('edgeThreshold', 10)
            sigma = kwargs.get('sigma', 1.6)
            
            return cv2.SIFT_create(
                nfeatures=nfeatures,
                nOctaveLayers=nOctaveLayers,
                contrastThreshold=contrastThreshold,
                edgeThreshold=edgeThreshold,
                sigma=sigma
            )
            
        elif self.detector_type == 'ORB':
            nfeatures = kwargs.get('nfeatures', 1000)
            scaleFactor = kwargs.get('scaleFactor', 1.2)
            nlevels = kwargs.get('nlevels', 8)
            edgeThreshold = kwargs.get('edgeThreshold', 31)
            
            return cv2.ORB_create(
                nfeatures=nfeatures,
                scaleFactor=scaleFactor,
                nlevels=nlevels,
                edgeThreshold=edgeThreshold
            )
            
        elif self.detector_type == 'AKAZE':
            descriptor_type = kwargs.get('descriptor_type', cv2.AKAZE_DESCRIPTOR_MLDB)
            descriptor_size = kwargs.get('descriptor_size', 0)
            descriptor_channels = kwargs.get('descriptor_channels', 3)
            threshold = kwargs.get('threshold', 0.001)
            
            return cv2.AKAZE_create(
                descriptor_type=descriptor_type,
                descriptor_size=descriptor_size,
                descriptor_channels=descriptor_channels,
                threshold=threshold
            )
            
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect keypoints and compute descriptors
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
            logger.info(f"Detected {len(keypoints)} keypoints using {self.detector_type}")
            return keypoints, descriptors
            
        except Exception as e:
            logger.error(f"Feature detection failed: {e}")
            return [], None
    
    def detect_multiple_images(self, images: List[np.ndarray]) -> List[Tuple]:
        """
        Detect features from multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of (keypoints, descriptors) tuples
        """
        features_data = []
        
        for i, img in enumerate(images):
            keypoints, descriptors = self.detect_and_compute(img)
            features_data.append((keypoints, descriptors))
            logger.info(f"Image {i+1}: {len(keypoints)} features detected")
            
        return features_data
    
    def visualize_keypoints(self, image: np.ndarray, keypoints: List, 
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected keypoints
        
        Args:
            image: Input image
            keypoints: Detected keypoints
            save_path: Optional path to save visualization
            
        Returns:
            Image with keypoints drawn
        """
        img_with_kp = cv2.drawKeypoints(
            image, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        if save_path:
            cv2.imwrite(save_path, img_with_kp)
            logger.info(f"Keypoints visualization saved to {save_path}")
            
        return img_with_kp
    
    def get_detector_info(self) -> dict:
        """
        Get information about current detector
        
        Returns:
            Dictionary with detector information
        """
        info = {
            'type': self.detector_type,
            'parameters': {}
        }
        
        if self.detector_type == 'SIFT':
            info['parameters'] = {
                'nfeatures': self.detector.getNFeatures(),
                'nOctaveLayers': self.detector.getNOctaveLayers(),
                'contrastThreshold': self.detector.getContrastThreshold(),
                'edgeThreshold': self.detector.getEdgeThreshold(),
                'sigma': self.detector.getSigma()
            }
        elif self.detector_type == 'ORB':
            info['parameters'] = {
                'nfeatures': self.detector.getMaxFeatures(),
                'scaleFactor': self.detector.getScaleFactor(),
                'nlevels': self.detector.getNLevels(),
                'edgeThreshold': self.detector.getEdgeThreshold()
            }
        elif self.detector_type == 'AKAZE':
            info['parameters'] = {
                'threshold': self.detector.getThreshold(),
                'nOctaves': self.detector.getNOctaves(),
                'nOctaveLayers': self.detector.getNOctaveLayers()
            }
            
        return info


def compare_detectors(image: np.ndarray, detector_types: List[str] = None) -> dict:
    """
    Compare different feature detectors on the same image
    
    Args:
        image: Input image
        detector_types: List of detector types to compare
        
    Returns:
        Dictionary with comparison results
    """
    if detector_types is None:
        detector_types = ['SIFT', 'ORB', 'AKAZE']
    
    results = {}
    
    for detector_type in detector_types:
        try:
            detector = FeatureDetector(detector_type)
            keypoints, descriptors = detector.detect_and_compute(image)
            
            results[detector_type] = {
                'num_keypoints': len(keypoints),
                'descriptor_shape': descriptors.shape if descriptors is not None else None,
                'detector_info': detector.get_detector_info()
            }
            
        except Exception as e:
            logger.error(f"Failed to test {detector_type}: {e}")
            results[detector_type] = {'error': str(e)}
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test with sample image
    import matplotlib.pyplot as plt
    
    # Load test image
    test_image = cv2.imread('../image/1.jpg', cv2.IMREAD_GRAYSCALE)
    
    if test_image is not None:
        # Compare different detectors
        comparison = compare_detectors(test_image)
        
        print("Feature Detector Comparison:")
        print("=" * 50)
        for detector_type, result in comparison.items():
            if 'error' not in result:
                print(f"{detector_type}: {result['num_keypoints']} keypoints")
                print(f"Descriptor shape: {result['descriptor_shape']}")
            else:
                print(f"{detector_type}: Error - {result['error']}")
        
        # Visualize SIFT features
        sift_detector = FeatureDetector('SIFT')
        keypoints, _ = sift_detector.detect_and_compute(test_image)
        img_with_kp = sift_detector.visualize_keypoints(
            test_image, keypoints, 'results/sift_keypoints.jpg'
        )
        
        print(f"\nSIFT keypoints visualization saved!")
    else:
        print("Test image not found!") 