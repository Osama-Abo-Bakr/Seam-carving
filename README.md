
# Seam Carving in Computer Vision

Seam carving is a content-aware image resizing technique that intelligently removes or adds seams (paths of least importance) to resize an image while preserving its most important features. This method was introduced by Shai Avidan and Ariel Shamir in 2007.

## How Seam Carving Works

- **Energy Map Calculation:**
    - The first step is to compute an energy map of the image, which highlights the importance of each pixel. Common methods to calculate the energy map include gradient magnitude (using operators like Sobel or Scharr), entropy maps, or saliency maps.
- **Seam Identification:**
    - A seam is a connected path of pixels from top to bottom (vertical seam) or left to right (horizontal seam) that has the lowest energy. This path is identified using dynamic programming, which efficiently finds the path with the minimum energy cost.
- **Seam Removal or Insertion:**
    - Once the seam is identified, it can be removed to reduce the image size or duplicated to increase the image size.

## Sample Code

Here's a Python example using OpenCV and NumPy to perform seam carving:

```python
import cv2
import numpy as np

def calculate_energy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(sobel_x) + np.abs(sobel_y)
    return energy

def find_seam(energy):
    rows, cols = energy.shape
    seam = np.zeros(rows, dtype=np.uint32)
    cost = energy.copy()
    for i in range(1, rows):
        for j in range(cols):
            min_cost = cost[i-1, j]
            if j > 0:
                min_cost = min(min_cost, cost[i-1, j-1])
            if j < cols - 1:
                min_cost = min(min_cost, cost[i-1, j+1])
            cost[i, j] += min_cost
    seam[rows-1] = np.argmin(cost[rows-1])
    for i in range(rows-2, -1, -1):
        prev_x = seam[i+1]
        min_cost = cost[i, prev_x]
        if prev_x > 0 and cost[i, prev_x-1] < min_cost:
            min_cost = cost[i, prev_x-1]
            seam[i] = prev_x - 1
        elif prev_x < cols - 1 and cost[i, prev_x+1] < min_cost:
            min_cost = cost[i, prev_x+1]
            seam[i] = prev_x + 1
        else:
            seam[i] = prev_x
    return seam

def remove_seam(image, seam):
    rows, cols, _ = image.shape
    output = np.zeros((rows, cols-1, 3), dtype=np.uint8)
    for i in range(rows):
        col = seam[i]
        output[i, :, 0] = np.delete(image[i, :, 0], col)
        output[i, :, 1] = np.delete(image[i, :, 1], col)
        output[i, :, 2] = np.delete(image[i, :, 2], col)
    return output

def seam_carve(image, num_seams):
    for _ in range(num_seams):
        energy = calculate_energy(image)
        seam = find_seam(energy)
        image = remove_seam(image, seam)
    return image

# Load image
image = cv2.imread('input.jpg')
num_seams = 50
output = seam_carve(image, num_seams)

# Save the result
cv2.imwrite('output.jpg', output)
