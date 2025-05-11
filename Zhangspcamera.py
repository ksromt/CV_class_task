import numpy as np
import cv2 as cv
import glob
import os

# Function to draw the box on the image
def draw_box(img, imgpts):
    # Draw base in green
    img = cv.line(img, tuple(imgpts[0][0]), tuple(imgpts[1][0]), (0, 255, 0), 3)
    img = cv.line(img, tuple(imgpts[0][0]), tuple(imgpts[2][0]), (0, 255, 0), 3)
    img = cv.line(img, tuple(imgpts[1][0]), tuple(imgpts[4][0]), (0, 255, 0), 3)
    img = cv.line(img, tuple(imgpts[2][0]), tuple(imgpts[4][0]), (0, 255, 0), 3)
    
    # Draw heights in blue
    img = cv.line(img, tuple(imgpts[0][0]), tuple(imgpts[3][0]), (255, 0, 0), 3)
    img = cv.line(img, tuple(imgpts[1][0]), tuple(imgpts[5][0]), (255, 0, 0), 3)
    img = cv.line(img, tuple(imgpts[2][0]), tuple(imgpts[6][0]), (255, 0, 0), 3)
    img = cv.line(img, tuple(imgpts[4][0]), tuple(imgpts[7][0]), (255, 0, 0), 3)
    
    # Draw top layer in red
    img = cv.line(img, tuple(imgpts[3][0]), tuple(imgpts[5][0]), (0, 0, 255), 3)
    img = cv.line(img, tuple(imgpts[3][0]), tuple(imgpts[6][0]), (0, 0, 255), 3)
    img = cv.line(img, tuple(imgpts[5][0]), tuple(imgpts[7][0]), (0, 0, 255), 3)
    img = cv.line(img, tuple(imgpts[6][0]), tuple(imgpts[7][0]), (0, 0, 255), 3)
    
    return img

# Configurable parameters - adjust based on your chessboard
# Try different values until detection succeeds
CHESS_BOARD_SIZE = (8, 9)  # Number of inner corners on the chessboard (width, height)
SQUARE_SIZE = 1.8  # Actual size of each square (cm)
IMAGE_PATTERNS = ['chessboard/*.jpg', 'chessboard/*.png', 'chessboard/*.jpeg']  # Support multiple image formats
USE_ADVANCED_DETECTION = True  # Use more advanced detection methods

# Create debug folder to save debug images
debug_dir = 'debug_images'
os.makedirs(debug_dir, exist_ok=True)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((CHESS_BOARD_SIZE[0] * CHESS_BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESS_BOARD_SIZE[0], 0:CHESS_BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Get all supported images
images = []
for pattern in IMAGE_PATTERNS:
    images.extend(glob.glob(pattern))
images = sorted(list(set(images)))  # Remove duplicates and sort
print(f"Found {len(images)} images: {images}")

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Failed to load image: {fname}")
        continue
    
    # Save original image for debugging
    cv.imwrite(os.path.join(debug_dir, f"original_{os.path.basename(fname)}"), img)
    
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Adjust image to enhance contrast
    gray = cv.equalizeHist(gray)
    cv.imwrite(os.path.join(debug_dir, f"enhanced_contrast_{os.path.basename(fname)}"), gray)
    
    found = False
    
    if USE_ADVANCED_DETECTION:
        # Try advanced chessboard detection method
        try:
            # First try findChessboardCornersSB method (OpenCV 4.x)
            ret, corners = cv.findChessboardCornersSB(gray, CHESS_BOARD_SIZE, None)
            if ret:
                found = True
                print(f"Successfully found chessboard corners using advanced method: {fname}")
        except:
            # If advanced method not available or fails, do nothing and continue with basic method
            pass
    
    if not found:
        # Use basic method
        # Try different chessboard detection flag combinations
        flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv.findChessboardCorners(gray, CHESS_BOARD_SIZE, flags)
        
        # If detection fails, try with more flags
        if not ret:
            flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FILTER_QUADS
            ret, corners = cv.findChessboardCorners(gray, CHESS_BOARD_SIZE, flags)
            
        if ret:
            found = True
            print(f"Successfully found chessboard corners using basic method: {fname}")
    
    # If chessboard corners were found
    if found:
        objpoints.append(objp)
        
        # Use cornerSubPix to refine corner position precision
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw detected corners
        img_with_corners = img.copy()
        cv.drawChessboardCorners(img_with_corners, CHESS_BOARD_SIZE, corners2, ret)
        
        # Save image with corners for debugging
        cv.imwrite(os.path.join(debug_dir, f"corner_detection_{os.path.basename(fname)}"), img_with_corners)
        
        # Display image
        cv.imshow('Chessboard Corners', img_with_corners)
        print(f"Displaying corners for {fname} - press any key to continue...")
        cv.waitKey(0)
    else:
        print(f"Could not find chessboard corners in {fname}. Try adjusting CHESS_BOARD_SIZE parameter or check image quality.")
        # Display the current image to help user understand why detection failed
        cv.imshow('Failed Detection Image', gray)
        print("Displaying failed detection image - press any key to continue...")
        cv.waitKey(0)

# Only perform camera calibration when enough data points are collected
if len(objpoints) > 0:
    print(f"Performing camera calibration using {len(objpoints)} images...")
    
    # Ensure we have a grayscale image for image dimensions
    if 'gray' not in locals():
        img = cv.imread(images[0])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Calculate camera calibration parameters
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Display results
    print("\nCamera matrix (intrinsic parameters):")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)
    
    # Print extrinsic parameters for each successful image
    print("\nExtrinsic Parameters for each image:")
    for i, fname in enumerate(images[:len(objpoints)]):
        print(f"\nImage: {fname}")
        print(f"Rotation Vector (rvec):")
        print(rvecs[i])
        print(f"Translation Vector (tvec):")
        print(tvecs[i])
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print("\nTotal mean reprojection error: {}".format(mean_error / len(objpoints)))
    print("Lower error means more accurate calibration. An error below 1 pixel is typically considered a good result.")
    
    # Save calibration results
    np.savez('camera_calibration.npz', 
             camera_matrix=mtx, 
             dist_coeffs=dist, 
             rvecs=rvecs, 
             tvecs=tvecs)
    
    # Draw 3D objects on the images
    # Define a simple 3D box
    axis = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3],
                        [3, 3, 0], [3, 0, -3], [0, 3, -3], [3, 3, -3]]) * SQUARE_SIZE
    
    for i, fname in enumerate(images[:len(objpoints)]):
        img = cv.imread(fname)
        if img is None:
            continue
        
        # Project 3D points to the image plane
        imgpts, jac = cv.projectPoints(axis, rvecs[i], tvecs[i], mtx, dist)
        imgpts = imgpts.astype(int)
        
        # Draw the box
        img_with_box = draw_box(img.copy(), imgpts)
        
        # Save image with 3D box
        cv.imwrite(os.path.join(debug_dir, f"with_3D_box_{os.path.basename(fname)}"), img_with_box)
        
        # Display image with 3D box
        cv.imshow(f'3D Box on {fname}', img_with_box)
        print(f"Displaying 3D box for {fname} - press any key to continue...")
        cv.waitKey(0)
else:
    print("No calibration data was collected. Please check your images and chessboard parameters.")

cv.destroyAllWindows()
print("Camera calibration process completed.")