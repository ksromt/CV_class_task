"""
Computer Vision Image Stitching Package
Author: AI Assistant
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@assistant.com"

from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher  
from .transform_estimator import TransformEstimator
from .image_warper import ImageWarper
from .image_blender import ImageBlender

__all__ = [
    'FeatureDetector',
    'FeatureMatcher', 
    'TransformEstimator',
    'ImageWarper',
    'ImageBlender'
] 