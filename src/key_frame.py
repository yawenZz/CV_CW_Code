import math
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def extract_frames(file_name, file_path):
    """
    Extract frames from the video file at every `stride` interval, which will be used for creating the panorama.
    :param file_name: Name of the video file (without extension).
    :param file_path: Path of the video file.
    :return: A list of extracted images.
    """
    # check and create folders to store the outputs
    key_frame_folder = '../keyframes/' + file_name + '/'
    results_folder = '../results/'
    os.makedirs(key_frame_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # construct VideoCapture object for frame-by-frame stream
    print(f"[INFO] loading video {file_path}...")
    video_cap = cv2.VideoCapture(file_path)

    # count total number of frames in the video
    total_frame = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'[INFO] Total number of frames in {file_name}.mp4: {total_frame}')

    # Read frames and save every `stride`-th frame
    stride = 40
    frame_count = 0
    extracted_frames = []   # list to store extracted frames

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        # Save frame if it's every `stride`-th frame
        if frame_count % stride == 0:
            print(f'[INFO] Frame {format(frame_count)} captured...')
            cv2.imwrite(key_frame_folder + 'frame' + str(frame_count) + '.jpg', frame)
            frame_cyl = cylindrical_wrap_nearest_neighbor(frame)
            frame_processed = crop_black(frame_cyl)
            extracted_frames.append(frame_processed)
        frame_count += 1

    video_cap.release()  # release video capture

    return extracted_frames


def nearest_neighbor_interpolation(img, x, y):
    x1 = int(round(x))
    y1 = int(round(y))

    # Ensure the coordinates are within the image bounds
    x1 = max(0, min(x1, img.shape[1] - 1))
    y1 = max(0, min(y1, img.shape[0] - 1))

    return img[y1, x1]


def cylindrical_wrap_nearest_neighbor(img, fov=20):
    height, width, depth = img.shape
    focal_length = width / (2 * math.tan(math.radians(fov)))

    center_x = width / 2
    center_y = height / 2

    cylinder = np.zeros_like(img)

    for j in range(height):
        for i in range(width):
            theta = (i - center_x) / focal_length
            point_x = int(focal_length * np.tan(theta) + center_x)
            point_y = int((j - center_y) / np.cos(theta) + center_y)

            if 0 <= point_x < width and 0 <= point_y < height:
                for k in range(depth):
                    cylinder[j, i, k] = nearest_neighbor_interpolation(img[:, :, k], point_x, point_y)

    # plt.figure()
    # plt.imshow(cylinder)
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    return cylinder


def crop_black(img):
    """
    Crop off the black edges using thresholding.
    :param img: image with black edges
    :return: cropped image
    """
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours on the binary image
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_rect = (0, 0, 0, 0)

    # Iterate through contours to find the largest one
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        deltaHeight = h - y
        deltaWidth = w - x

        area = deltaHeight * deltaWidth

        if area > max_area and deltaHeight > 0 and deltaWidth > 0:
            max_area = area
            best_rect = (x, y, w, h)

    # Crop the image using the bounding rectangle
    if max_area > 0:
        img_crop = img[best_rect[1]:best_rect[1] + best_rect[3], best_rect[0]:best_rect[0] + best_rect[2]]
    else:
        img_crop = img.copy()

    return img_crop
