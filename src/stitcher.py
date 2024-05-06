import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


def stitch(img_right, img_left):
    """
    The main method to stitch images.
    :param img_right: image containing previous stitching results
    :param img_left: image to be added
    :return: the stitched image
    """
    # calculate the homography matrix
    H, _ = calculate_homography(img_right, img_left)

    # get heights and weights of two images
    h1, w1 = img_right.shape[0:2]
    h2, w2 = img_left.shape[0:2]

    # get 4 corners of two images: top-left, bottom-left, top-right, bottom-right
    img_right_pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img_left_pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # apply homography to corners of img_right
    img_right_pts_ = cv2.perspectiveTransform(img_right_pts, H)

    # get the coordinate range of output image
    output_pts = np.concatenate((img_right_pts_, img_left_pts), axis=0)
    [xmin, ymin] = np.int64(output_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int64(output_pts.max(axis=0).ravel() + 0.5)

    # translation reference: https://stackoverflow.com/a/20355545
    translation_mtx = np.array([[1, 0, -xmin],
                                [0, 1, -ymin],
                                [0, 0, 1]])

    # wrap the perspective of right image
    output_img = cv2.warpPerspective(img_right, translation_mtx.dot(H), (xmax - xmin, ymax - ymin))

    # --- direct blending ---
    # add left image to the transformed right image
    output_img[-ymin:h2 - ymin, -xmin:w2 - xmin] = img_left

    # # crop the image
    # output_img = crop_image(output_img)
    # plt.figure()
    # plt.imshow(output_img)
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # --- alpha blending ---
    # # Add left image to the transformed right image with alpha blending
    # mask_start = 0  # The left index of blending
    # mask_end = img_left.shape[1]  # The right index of blending
    # for i in range(img_left.shape[1]):
    #     if output_img[0, i, :].any() != 0:
    #         mask_start = i
    #         break
    #
    # output_img = alpha_blend(output_img, img_left, mask_start, mask_end)

    return output_img


def get_keypoints_and_descriptors(image):
    """
    Find keypoints and descriptors of @image using SIFT Algorithm.
    :param image:
    :return: keypoints, descriptors of img
    """
    # convert colour img into gray img
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # init SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints, features = sift.detectAndCompute(image, None)

    return keypoints, features


def match_features(feature_right, feature_left, match_ratio=0.7):
    """
    Match features between two images using BruteForce Algorithm.
    :param feature_right:
    :param feature_left:
    :param match_ratio: Lowe's ratio test.
    :return: matches
    """
    # init Brute Force Matcher
    bf = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = bf.knnMatch(feature_right, feature_left, k=2)

    # store good matches using ratio test
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < match_ratio * m2.distance:
            good_matches.append(m1)

    if len(good_matches) > 4:
        return good_matches
    else:
        raise Exception("[INFO] No enough good matches.")


def calculate_homography(src_img, dst_img):
    """
    Calculate Homography matrix using RANSAC algorithm.
    :param src_img: image wrapped by homography
    :param dst_img: image chosen as pivot
    :return: Homography matrix, matchesMask
    """
    src_kpts, src_features = get_keypoints_and_descriptors(src_img)
    dst_kpts, dst_features = get_keypoints_and_descriptors(dst_img)

    good_matches = match_features(src_features, dst_features)

    # formalize as matrices
    src_pts = np.float32([src_kpts[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_kpts[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # computer homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # img3 = draw_matches(src_img, src_kpts, dst_img, dst_kpts, good_matches, matchesMask)
    # plt.figure()
    # plt.imshow(img3, 'gray')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    return H, matchesMask


def draw_keypoints(img, kpts):
    """
    Helper function to visualize found keypoints.
    """
    output = img
    cv2.drawKeypoints(img, kpts, output)
    return output


def draw_matches(src_img, src_kpts, dst_img, dst_kpts, matches, matchesMask):
    """
    Helper function to visualize found matches.
    """
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    return cv2.drawMatches(src_img, src_kpts, dst_img, dst_kpts, matches, None, **draw_params)


def alpha_blend(img_output, img_left, mask_start, mask_end):
    """
    Perform alpha blending between the overlapping region of img_output and img_left.
    :param img_output: Image where img_left will be blended.
    :param img_left: Image to be blended onto img_output.
    :param mask_start: Start index of blending region.
    :param mask_end: End index of blending region.
    :return: Blended image.
    """
    for i in range(img_left.shape[0]):
        for j in range(mask_start, mask_end):  # Iterate within blending region
            if j < img_output.shape[1]:  # Ensure within output image bounds
                if img_left[i, j, :].any() != 0:  # Check if pixel is not black
                    if img_output[i, j, :].any() != 0:  # Check if pixel in output image is not black
                        alpha = (j - mask_start) / (mask_end - mask_start)  # Calculate alpha value
                        img_output[i, j] = img_left[i, j] * (1 - alpha) + img_output[i, j] * alpha  # Alpha blending
                    else:
                        img_output[i, j] = img_left[i, j]  # Direct assignment if pixel in output image is black
    return img_output
