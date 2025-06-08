import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class EpipolarGeometry:
    def __init__(self, img1_path, img2_path):
        """
        Initialize with two images of the same scene from different viewpoints
        """
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        
        # Initialize feature detector
        self.sift = cv2.SIFT_create()
        
        # Store keypoints and matches
        self.kp1 = None
        self.kp2 = None
        self.matches = None
        self.fundamental_matrix = None
        
    def detect_and_match_features(self):
        """
        Detect SIFT features and match between two images
        """
        # Detect keypoints and descriptors
        self.kp1, des1 = self.sift.detectAndCompute(self.img1_gray, None)
        self.kp2, des2 = self.sift.detectAndCompute(self.img2_gray, None)
        
        # Match features using FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        self.matches = good_matches
        print(f"Found {len(good_matches)} good matches")
        
    def estimate_fundamental_matrix(self):
        """
        Estimate fundamental matrix using RANSAC
        """
        if self.matches is None:
            self.detect_and_match_features()
        
        # Extract matched points
        pts1 = np.float32([self.kp1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        pts2 = np.float32([self.kp2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        
        # Estimate fundamental matrix using RANSAC
        self.fundamental_matrix, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99
        )
        
        # Filter inliers
        self.inlier_matches = [self.matches[i] for i in range(len(mask)) if mask[i]]
        print(f"Fundamental matrix estimated with {len(self.inlier_matches)} inliers")
        
        return self.fundamental_matrix
    
    def draw_epipolar_lines(self, img1_pts, img2_pts, img1, img2):
        """
        Draw epipolar lines for given point correspondences
        """
        # Find epilines corresponding to points in img1, draw on img2
        lines1 = cv2.computeCorrespondEpilines(img2_pts.reshape(-1, 1, 2), 2, self.fundamental_matrix)
        lines1 = lines1.reshape(-1, 3)
        img2_lines = self.draw_lines(img2.copy(), lines1, img2_pts, img1_pts)
        
        # Find epilines corresponding to points in img2, draw on img1
        lines2 = cv2.computeCorrespondEpilines(img1_pts.reshape(-1, 1, 2), 1, self.fundamental_matrix)
        lines2 = lines2.reshape(-1, 3)
        img1_lines = self.draw_lines(img1.copy(), lines2, img1_pts, img2_pts)
        
        return img1_lines, img2_lines
    
    def draw_lines(self, img, lines, pts1, pts2):
        """
        Draw epipolar lines on image
        """
        r, c = img.shape[:2]
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        
        for i, (line, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
            color = colors[i % len(colors)]
            
            # Calculate line endpoints
            x0, y0 = map(int, [0, -line[2]/line[1]])
            x1, y1 = map(int, [c, -(line[2]+line[0]*c)/line[1]])
            
            # Draw epipolar line
            cv2.line(img, (x0, y0), (x1, y1), color, 2)
            
            # Draw the corresponding point
            cv2.circle(img, tuple(map(int, pt1)), 8, color, -1)
        
        return img
    
    def visualize_matches_and_epipolar_lines(self, num_points=8):
        """
        Visualize feature matches and epipolar lines
        """
        if self.fundamental_matrix is None:
            self.estimate_fundamental_matrix()
        
        # Select a subset of good matches for visualization
        selected_matches = self.inlier_matches[:num_points]
        
        # Extract points
        pts1 = np.float32([self.kp1[m.queryIdx].pt for m in selected_matches])
        pts2 = np.float32([self.kp2[m.trainIdx].pt for m in selected_matches])
        
        # Draw epipolar lines
        img1_with_lines, img2_with_lines = self.draw_epipolar_lines(pts1, pts2, self.img1, self.img2)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original matches
        img_matches = cv2.drawMatches(
            self.img1, self.kp1, self.img2, self.kp2,
            selected_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        axes[0, 0].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Feature Matches')
        axes[0, 0].axis('off')
        
        # Image 1 with epipolar lines
        axes[0, 1].imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Image 1 with Epipolar Lines')
        axes[0, 1].axis('off')
        
        # Image 2 with epipolar lines
        axes[1, 0].imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Image 2 with Epipolar Lines')
        axes[1, 0].axis('off')
        
        # Show fundamental matrix
        axes[1, 1].text(0.1, 0.7, 'Fundamental Matrix:', fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.5, str(self.fundamental_matrix), fontsize=8, family='monospace')
        axes[1, 1].text(0.1, 0.3, f'Number of inliers: {len(self.inlier_matches)}', fontsize=10)
        axes[1, 1].text(0.1, 0.2, f'Total matches: {len(self.matches)}', fontsize=10)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return pts1, pts2
    
    def verify_epipolar_constraint(self, pts1, pts2):
        """
        Verify that corresponding points satisfy the epipolar constraint: x2^T * F * x1 = 0
        """
        if self.fundamental_matrix is None:
            print("Fundamental matrix not computed yet!")
            return
        
        # Convert to homogeneous coordinates
        pts1_h = np.column_stack([pts1, np.ones(len(pts1))])
        pts2_h = np.column_stack([pts2, np.ones(len(pts2))])
        
        # Calculate epipolar constraint for each point pair
        constraints = []
        for i in range(len(pts1)):
            constraint = pts2_h[i] @ self.fundamental_matrix @ pts1_h[i].T
            constraints.append(constraint)
        
        constraints = np.array(constraints)
        mean_error = np.mean(np.abs(constraints))
        
        print(f"Epipolar constraint verification:")
        print(f"Mean absolute error: {mean_error:.6f}")
        print(f"Individual errors: {constraints}")
        print("Values close to 0 indicate good correspondence along epipolar lines")
        
        return constraints

def main():
    """
    Main function to demonstrate epipolar geometry
    """
    print("=== Epipolar Geometry Demonstration ===")
    print("Using existing images for demonstration...")
    
    # Try to use existing images first
    try:
        # First try with user-provided images
        if os.path.exists('image1.jpg') and os.path.exists('image2.jpg'):
            print("Using user-provided images: image1.jpg and image2.jpg")
            epipolar = EpipolarGeometry('image1.jpg', 'image2.jpg')
        # If not available, use sample images from lec2_image directory
        elif os.path.exists('lec2_image/1.jpg') and os.path.exists('lec2_image/planar_image.jpg'):
            print("Using sample images from lec2_image directory")
            epipolar = EpipolarGeometry('lec2_image/1.jpg', 'lec2_image/planar_image.jpg')
        else:
            print("No suitable image pairs found!")
            print("Please provide two images of the same scene from different viewpoints:")
            print("- Name them 'image1.jpg' and 'image2.jpg'")
            print("- Place them in the current directory")
            return
        
        # Detect and match features
        epipolar.detect_and_match_features()
        
        # Estimate fundamental matrix
        F = epipolar.estimate_fundamental_matrix()
        print(f"Fundamental Matrix:\n{F}")
        
        # Visualize results
        pts1, pts2 = epipolar.visualize_matches_and_epipolar_lines(num_points=8)
        
        # Verify epipolar constraint
        epipolar.verify_epipolar_constraint(pts1, pts2)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have two images of the same scene from different viewpoints")

if __name__ == "__main__":
    main()
