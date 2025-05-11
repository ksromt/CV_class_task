import cv2
import numpy as np
import cv2.ximgproc as xip

'''
Bilateral and guided image filters
'''
# # #add noise
# # img = cv2.imread('lec2_image/1.jpg')
# # noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
# # noisy_img = cv2.add(img, noise)
# # cv2.imwrite('lec2_image/noisy_image2.jpg', noisy_img)
#
# img = cv2.imread('lec2_image/noisy_image.jpg')
#
# # Bilateral
#
#
# bilateral1 = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
#
# bilateral2 = cv2.bilateralFilter(img, d=15, sigmaColor=150, sigmaSpace=150)
#
# bilateral3 = cv2.bilateralFilter(img, d=25, sigmaColor=200, sigmaSpace=200)
#
# # guided
# guided1 = xip.guidedFilter(guide=img, src=img, radius=8, eps=0.01)
#
# guided2 = xip.guidedFilter(guide=img, src=img, radius=16, eps=0.1**2 * 255**2)
#
# guided3 = xip.guidedFilter(guide=img, src=img, radius=32, eps=2000)
#
# cv2.imshow('Original', img)
# cv2.imshow('Bilateral1', bilateral1)
# cv2.imshow('Bilateral2', bilateral2)
# cv2.imshow('Bilateral3', bilateral3)
# cv2.imshow('Guided1', guided1)
# cv2.imshow('Guided2', guided2)
# cv2.imshow('Guided3', guided3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
Image warping
'''

import cv2
import numpy as np

points = []  # save four corner

def mouse_click(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Select 4 Corners', img)
        if len(points) == 4:
            print("Please click in order: top-left, top-right, bottom-left, bottom-right")

# img = cv2.imread('lec2_image/planar_image.jpg')
img = cv2.imread('lec2_image/non-planar_image.jpg')
cv2.namedWindow('Select 4 Corners')
print("Please click the four corner points in order: top-left, top-right, bottom-left, bottom-right")
cv2.setMouseCallback('Select 4 Corners', mouse_click)

while len(points) < 4:
    cv2.imshow('Select 4 Corners', img)
    cv2.waitKey(1)

# Convert points to numpy array
points_array = np.float32(points)

# Order points in clockwise manner:
# Sort by y-coordinate (top/bottom)
sorted_by_y = points_array[points_array[:, 1].argsort()]
# Get top and bottom points
top_points = sorted_by_y[:2]
bottom_points = sorted_by_y[2:]

# Sort top points by x-coordinate (left to right)
top_left = top_points[np.argmin(top_points[:, 0])]
top_right = top_points[np.argmax(top_points[:, 0])]

# Sort bottom points by x-coordinate (left to right)
bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]

# Set ordered points
pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
cv2.destroyAllWindows()

# Calculate aspect ratio more precisely
# Calculate average length of two sides to get a more accurate aspect ratio
width_top = np.linalg.norm(top_right - top_left)
width_bottom = np.linalg.norm(bottom_right - bottom_left)
avg_width = (width_top + width_bottom) / 2

height_left = np.linalg.norm(bottom_left - top_left)
height_right = np.linalg.norm(bottom_right - top_right)
avg_height = (height_left + height_right) / 2

# Maintain aspect ratio while determining appropriate dimensions
if avg_width > avg_height:
    # Width is the dominant factor
    max_width = 500  # Set reasonable maximum width
    max_height = int(max_width * (avg_height / avg_width))
else:
    # Height is the dominant factor
    max_height = 900  # Set reasonable maximum height
    max_width = int(max_height * (avg_width / avg_height))

# Define destination points for the rectangle
pts2 = np.float32([
    [0, 0],               # top-left
    [max_width, 0],       # top-right
    [0, max_height],      # bottom-left
    [max_width, max_height]  # bottom-right
])

# Execute perspective transformation
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (max_width, max_height))

cv2.imshow('Original', img)
cv2.imshow('Warped', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

