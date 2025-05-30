"""
Feature Matching Module for Image Stitching
Implements brute force and FLANN-based matching with NNDR filtering
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class FeatureMatcher:
    """
    Feature matcher class supporting multiple matching strategies
    """
    
    def __init__(self, matcher_type: str = 'BF', nndr_ratio: float = 0.65):
        """
        Initialize feature matcher
        
        Args:
            matcher_type: Type of matcher ('BF' or 'FLANN')
            nndr_ratio: Nearest neighbor distance ratio threshold (降低到0.65)
        """
        self.matcher_type = matcher_type
        self.nndr_ratio = nndr_ratio
        self.matcher = None
        
    def _create_matcher(self, descriptor_type: str = 'float'):
        """
        Create specific matcher instance
        
        Args:
            descriptor_type: Type of descriptor ('float' for SIFT/AKAZE, 'binary' for ORB)
            
        Returns:
            OpenCV matcher object
        """
        if self.matcher_type == 'BF':
            if descriptor_type == 'binary':
                # For ORB descriptors
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                # For SIFT/AKAZE descriptors
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                
        elif self.matcher_type == 'FLANN':
            if descriptor_type == 'binary':
                # FLANN parameters for ORB
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                  table_number=6,
                                  key_size=12,
                                  multi_probe_level=1)
            else:
                # FLANN parameters for SIFT/AKAZE
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unsupported matcher type: {self.matcher_type}")
    
    def match_features(self, features1: Tuple, features2: Tuple) -> List[cv2.DMatch]:
        """
        Match features between two images
        
        Args:
            features1: Tuple of (keypoints, descriptors) for first image
            features2: Tuple of (keypoints, descriptors) for second image
            
        Returns:
            List of good matches after NNDR filtering
        """
        kp1, des1 = features1
        kp2, des2 = features2
        
        if des1 is None or des2 is None:
            logger.warning("One or both descriptor arrays are None")
            return []
        
        # Determine descriptor type
        descriptor_type = 'binary' if des1.dtype == np.uint8 else 'float'
        
        # Create matcher if not exists or type changed
        if self.matcher is None:
            self._create_matcher(descriptor_type)
        
        try:
            # Perform KNN matching
            matches = self.matcher.knnMatch(des1, des2, k=2)
            
            # Apply NNDR (Nearest Neighbor Distance Ratio) test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.nndr_ratio * n.distance:
                        good_matches.append(m)
            
            logger.info(f"Found {len(good_matches)} good matches out of {len(matches)} total matches")
            return good_matches
            
        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            return []
    
    def get_matched_points(self, keypoints1: List, keypoints2: List, 
                          matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract matched point coordinates
        
        Args:
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            matches: List of DMatch objects
            
        Returns:
            Tuple of (src_points, dst_points) as numpy arrays
        """
        if len(matches) == 0:
            return None, None
        
        # Extract matched keypoint coordinates
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        return src_pts, dst_pts
    
    def filter_matches_by_distance(self, matches: List[cv2.DMatch], 
                                  distance_threshold: float = None) -> List[cv2.DMatch]:
        """
        Filter matches by distance threshold
        
        Args:
            matches: List of matches
            distance_threshold: Maximum allowed distance (if None, use median + std)
            
        Returns:
            Filtered matches
        """
        if not matches:
            return []
        
        distances = [m.distance for m in matches]
        
        if distance_threshold is None:
            # Use adaptive threshold based on statistics
            median_dist = np.median(distances)
            std_dist = np.std(distances)
            distance_threshold = median_dist + std_dist
        
        filtered_matches = [m for m in matches if m.distance <= distance_threshold]
        
        logger.info(f"Distance filtering: {len(filtered_matches)}/{len(matches)} matches retained")
        return filtered_matches
    
    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray,
                         kp1: List, kp2: List, matches: List[cv2.DMatch],
                         max_matches: int = 50, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize feature matches between two images
        
        Args:
            img1, img2: Input images
            kp1, kp2: Keypoints for both images
            matches: List of matches
            max_matches: Maximum number of matches to display
            save_path: Optional path to save visualization
            
        Returns:
            Image with matches drawn
        """
        # Sort matches by distance and take the best ones
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        display_matches = sorted_matches[:max_matches]
        
        # Draw matches
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, display_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        if save_path:
            cv2.imwrite(save_path, match_img)
            logger.info(f"Match visualization saved to {save_path}")
        
        return match_img
    
    def compute_match_statistics(self, matches: List[cv2.DMatch]) -> dict:
        """
        Compute statistics for matches
        
        Args:
            matches: List of matches
            
        Returns:
            Dictionary with match statistics
        """
        if not matches:
            return {'num_matches': 0}
        
        distances = [m.distance for m in matches]
        
        stats = {
            'num_matches': len(matches),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances)
        }
        
        return stats
    
    def match_multiple_images(self, features_list: List[Tuple], 
                            reference_idx: int = 0) -> List[List[cv2.DMatch]]:
        """
        Match multiple images against a reference image
        
        Args:
            features_list: List of (keypoints, descriptors) tuples
            reference_idx: Index of reference image
            
        Returns:
            List of matches for each image pair
        """
        reference_features = features_list[reference_idx]
        all_matches = []
        
        for i, features in enumerate(features_list):
            if i == reference_idx:
                all_matches.append([])  # No self-matching
                continue
                
            matches = self.match_features(reference_features, features)
            all_matches.append(matches)
            
            logger.info(f"Image {reference_idx+1} <-> Image {i+1}: {len(matches)} matches")
        
        return all_matches


def create_track_indices(features_list: List[Tuple], matcher: FeatureMatcher) -> dict:
    """
    Create unique track indices for feature points across multiple images
    This implements step 3 of the assignment
    
    Args:
        features_list: List of (keypoints, descriptors) tuples
        matcher: FeatureMatcher instance
        
    Returns:
        Dictionary mapping image indices to track assignments
    """
    tracks = {}
    track_id = 0
    
    # Start with first image as reference
    reference_features = features_list[0]
    kp_ref, des_ref = reference_features
    
    # Initialize tracks for reference image
    tracks[0] = list(range(len(kp_ref)))
    track_id = len(kp_ref)
    
    # Match each subsequent image with reference
    for img_idx in range(1, len(features_list)):
        matches = matcher.match_features(reference_features, features_list[img_idx])
        kp_current, _ = features_list[img_idx]
        
        # Initialize track assignments for current image
        current_tracks = [-1] * len(kp_current)
        
        # Assign existing tracks to matched features
        for match in matches:
            ref_idx = match.queryIdx
            curr_idx = match.trainIdx
            
            # Assign the same track ID as reference
            if ref_idx < len(tracks[0]):
                current_tracks[curr_idx] = tracks[0][ref_idx]
        
        # Assign new track IDs to unmatched features
        for i in range(len(current_tracks)):
            if current_tracks[i] == -1:
                current_tracks[i] = track_id
                track_id += 1
        
        tracks[img_idx] = current_tracks
    
    logger.info(f"Created {track_id} unique feature tracks across {len(features_list)} images")
    return tracks


# Example usage and testing
if __name__ == "__main__":
    from feature_detector import FeatureDetector
    
    # Load test images
    img1 = cv2.imread('../image/1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../image/2.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img1 is not None and img2 is not None:
        # Detect features
        detector = FeatureDetector('SIFT')
        features1 = detector.detect_and_compute(img1)
        features2 = detector.detect_and_compute(img2)
        
        # Match features
        matcher = FeatureMatcher('BF', nndr_ratio=0.65)
        matches = matcher.match_features(features1, features2)
        
        # Compute statistics
        stats = matcher.compute_match_statistics(matches)
        print("Match Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Visualize matches
        kp1, _ = features1
        kp2, _ = features2
        match_img = matcher.visualize_matches(
            img1, img2, kp1, kp2, matches,
            max_matches=30, save_path='results/feature_matches.jpg'
        )
        
        print(f"Feature matching visualization saved!")
    else:
        print("Test images not found!") 