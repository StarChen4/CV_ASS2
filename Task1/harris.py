"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID):Xing Chen(u7725171)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.01

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()

# bw = plt.imread('0.png')
# bw = np.array(bw * 255, dtype=int)
# # computer x and y derivatives of image
# Ix = conv2(bw, dx)
# Iy = conv2(bw, dy)
#
# # generate a gaussian filter of given sigma
# g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
# # compute Iy^2 and smooth it with g filter
# Iy2 = conv2(np.power(Iy, 2), g)
# # compute Ix^2 and smooth it with g filter
# Ix2 = conv2(np.power(Ix, 2), g)
# # compute Ix * Iy and smooth it with g filter
# Ixy = conv2(Ix * Iy, g)

######################################################################
# Task: Compute the Harris Cornerness
######################################################################


def compute_harris_response(Ix2, Iy2, Ixy, k=0.05):
    """
    Compute the Harris corner response for an image.
    Parameters:
    - Ix2: numpy array, square of the gradient in the x-direction.
    - Iy2: numpy array, square of the gradient in the y-direction.
    - Ixy: numpy array, product of the gradients in the x and y directions.
    - k: float, Harris corner detection parameter, default is 0.05.

    Returns:
    - R: numpy array, Harris corner response of the image.
    """

    # Compute determinant and trace of the structure tensor
    detM = Ix2 * Iy2 - Ixy ** 2
    traceM = Ix2 + Iy2

    # Compute the Harris corner response
    R = detM - k * (traceM**2)
    return R


######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################


def non_max_suppression_thresholding(r, window_size=5, threshold_ratio=0.01):
    """
    Perform non-maximum suppression and thresholding to identify corner points in an image.
    Parameters:
    - r: numpy array, the Harris response image.
    - window_size: int, size of the non-maximum suppression window, default value is 5.
    - threshold_ratio: float, ratio used to determine the threshold, default value is 0.01.

    Returns:
    - nparray: numpy array representing the suppressed image with identified corners.
    """

    # get the height and width of R
    height, width = r.shape
    # the response matrix after suppression initialization
    corners_image = np.zeros((height, width), dtype=np.float32)
    # threshold calculation
    threshold = threshold_ratio * np.max(r)

    # iterate through each pixel
    for i in range(height):
        for j in range(width):
            # get the window centered at current pixel located at [i, j]
            start_row = max(0, i - window_size // 2)
            end_row = min(height, i + window_size // 2 + 1)
            start_col = max(0, j - window_size // 2)
            end_col = min(width, j + window_size // 2 + 1)
            # the window around this pixel
            window = r[start_row:end_row, start_col:end_col]

            # execute non-maximum suppression
            if r[i, j] == np.max(window) and r[i, j] > threshold:
                corners_image[i, j] = r[i, j]

    # convert the matrix into N*2
    corners = np.argwhere(corners_image > 0)
    return corners


# # compute harris response value
# R = compute_harris_response(Ix2, Iy2, Ixy, k=0.05)
# # apply suppression on R
# corners = non_max_suppression_thresholding(R)

######################################################################
# image test function
######################################################################
# images paths
test_image_array = ['./Harris-1.jpg', './Harris-2.jpg', './Harris-3.jpg', './Harris-4.jpg']


def test_harris_on_image(images_paths):
    """
    do harris corner detection on some test image and plot the result
    """

    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    # Process each image
    for idx, img_path in enumerate(images_paths):
        # Read and convert image to grayscale if necessary
        img = cv2.imread(img_path)
        # if the image is colored, convert to grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)

        # Compute x and y derivatives of image
        Ix = conv2(img, dx)
        Iy = conv2(img, dy)

        # Generate a Gaussian filter of given sigma
        g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)

        # Compute squares and products of derivatives, and smooth with Gaussian filter
        Ix2 = conv2(np.power(Ix, 2), g)
        Iy2 = conv2(np.power(Iy, 2), g)
        Ixy = conv2(Ix * Iy, g)

        # Compute Harris corner response
        R = compute_harris_response(Ix2, Iy2, Ixy, k=0.05)

        # Apply non-maximum suppression and thresholding
        corner_coords = non_max_suppression_thresholding(R)

        # Plot original image and overlay corners
        axes[idx].imshow(img, cmap='gray')
        axes[idx].scatter(corner_coords[:, 1], corner_coords[:, 0], s=40, facecolors='none', edgecolors='r', linewidths=1.5)
        axes[idx].set_title(f'Harris Corners in Image {idx+1}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


# my function to detect corner
test_harris_on_image(test_image_array)


def cv2_harris_on_image(images_paths):
    """
    use cv2 harris corner detection on some test image to compare the result
    """
    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    # Process each image
    for idx, img_path in enumerate(images_paths):
        # Read and convert image to grayscale if necessary
        img = cv2.imread(img_path)
        # if the image is colored, convert to grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)

        # compute harris cornerness using cv2
        harris_response = cv2.cornerHarris(img, blockSize=5, ksize=3, k=0.05)
        # Apply non-maximum suppression and thresholding
        corner_coords = non_max_suppression_thresholding(harris_response)

        # Plot original image and overlay corners
        axes[idx].imshow(img, cmap='gray')
        axes[idx].scatter(corner_coords[:, 1], corner_coords[:, 0], s=40, facecolors='none', edgecolors='r',
                          linewidths=1.5)
        axes[idx].set_title(f'Harris Corners in Image {idx + 1}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


# cv2 build-in function to detect corner
cv2_harris_on_image(test_image_array)


def inverse_warping(image, inv_transform, output_size):
    # create the output image template
    warped_image = np.zeros(output_size, dtype=np.float32)

    # get size of the output
    rows, cols = output_size[:2]

    # iterate through all pixels
    for i in range(rows):
        for j in range(cols):
            # find the corresponding position of given pixel using inverse transform matrix
            original_pos = np.dot(inv_transform, np.array([j, i, 1]))
            eps = 1e-10 # a very small value to prevent derive by 0
            orig_x, orig_y, _ = original_pos / (original_pos[2] + eps)  # normalization

            # bilinear interpolation
            # make sure it doesn't go beyond the outside
            if (0 <= orig_x < image.shape[1]-1) and (0 <= orig_y < image.shape[0]-1):
                x_floor, y_floor = np.floor([orig_x, orig_y]).astype(int)
                x_ceil, y_ceil = np.ceil([orig_x, orig_y]).astype(int)

                # get four closest pixels
                top_left = image[y_floor, x_floor]
                top_right = image[y_floor, x_ceil]
                bottom_left = image[y_ceil, x_floor]
                bottom_right = image[y_ceil, x_ceil]

                # compute interpolation value
                top = (x_ceil - orig_x) * top_left + (orig_x - x_floor) * top_right
                bottom = (x_ceil - orig_x) * bottom_left + (orig_x - x_floor) * bottom_right
                pixel_value = (y_ceil - orig_y) * top + (orig_y - y_floor) * bottom

                # set the pixel value
                warped_image[i, j] = pixel_value

    return warped_image


def cv2_inverse_warping(image, inv_transform, output_size):
    # Create the output image template
    warped_image = np.zeros(output_size, dtype=np.float32)

    # Get size of the output
    rows, cols = output_size[:2]

    # Prepare grid coordinates
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)

    # Generate map for remap
    for i in range(rows):
        for j in range(cols):
            # Apply the inverse transformation matrix
            original_pos = np.dot(inv_transform, np.array([j, i, 1]))
            map_x[i, j] = original_pos[0] / original_pos[2]  # x-coordinates
            map_y[i, j] = original_pos[1] / original_pos[2]  # y-coordinates

    # Use cv2.remap to perform the actual image warping using bilinear interpolation
    warped_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return warped_image


def rotate_and_detect(image_path, rotation_angles):
    """
    Load an image, perform rotation for given angles, and detect Harris corners.
    Args:
        image_path (str): Path to the input image.
        rotation_angles (list of int): List of angles in degrees to rotate the image clockwise.

    Returns:
        None: The function displays a matplotlib figure with subplots showing the rotated images
              and their detected Harris corners.
    """
    # Read the image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    # Create a figure for displaying the results
    fig, axes = plt.subplots(1, len(rotation_angles), figsize=(20, 5))
    fig.suptitle(f"Harris Corners after Rotation - {image_path}")

    for idx, angle in enumerate(rotation_angles):
        # Create the rotation matrix
        rows, cols = gray.shape
        center = (cols / 2, rows / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

        # Convert the rotation matrix to a 3x3 homogeneous transformation matrix
        homogeneous_matrix = np.eye(3)
        homogeneous_matrix[:2, :3] = rotation_matrix

        # Perform the inverse rotation
        # rotated_img = inverse_warping(gray, np.linalg.inv(homogeneous_matrix), gray.shape)
        rotated_img = cv2_inverse_warping(gray, np.linalg.inv(homogeneous_matrix), gray.shape)
        # rotated_img = cv2.warpPerspective(gray, np.linalg.inv(homogeneous_matrix), gray.shape)

        # Compute Harris corners on the rotated image
        Ix = conv2(rotated_img, dx)
        Iy = conv2(rotated_img, dy)
        g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
        Ix2 = conv2(np.power(Ix, 2), g)
        Iy2 = conv2(np.power(Iy, 2), g)
        Ixy = conv2(Ix * Iy, g)
        R = compute_harris_response(Ix2, Iy2, Ixy, k=0.05)
        corner_coords = non_max_suppression_thresholding(R)

        # Plot the rotated image and detected corners
        axes[idx].imshow(rotated_img, cmap='gray')
        axes[idx].scatter(corner_coords[:, 1], corner_coords[:, 0], s=20, facecolors='none', edgecolors='r', linewidths=1)
        axes[idx].set_title(f"Rotated by {angle}°")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


# rotate and detect
rotation_angles = [0, 90, 180, 270]
rotate_and_detect(test_image_array[0], rotation_angles)


def visualize_harris_response(image_path):
    # read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    # compute harris response and apply non_max_suppression_thresholding
    Ix = conv2(gray, dx)
    Iy = conv2(gray, dy)
    g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
    Ix2 = conv2(np.power(Ix, 2), g)
    Iy2 = conv2(np.power(Iy, 2), g)
    Ixy = conv2(Ix * Iy, g)
    R = compute_harris_response(Ix2, Iy2, Ixy, k=0.05)
    corners = non_max_suppression_thresholding(R)

    # subplot creation
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    # colored original image
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # plot gray image and corners
    ax[1].imshow(gray, cmap='gray')
    ax[1].plot(corners[:, 1], corners[:, 0], 'go', markersize=10, linewidth=1.5)
    ax[1].set_title('Gray Image with Corners')
    ax[1].axis('off')

    # plot heat map of harris scores
    ax[2].imshow(R, cmap='hot')
    ax[2].set_title('Harris Response Heatmap')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


visualize_harris_response('./Harris-5.jpg')
visualize_harris_response('./Harris-6.jpg')



