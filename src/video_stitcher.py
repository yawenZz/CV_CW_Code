import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QFont
from PyQt5.QtCore import Qt
import cv2
from stitcher import stitch
from key_frame import extract_frames


class VideoStitcherApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Video Stitcher')
        self.resize(800, 600)

        self.file_path_label = QLabel('No file selected.')
        self.file_path_label.setAlignment(Qt.AlignCenter)

        self.video1_button = QPushButton('Sample Video 1: library ')
        self.video1_button.clicked.connect(self.selectVideo1)
        self.video2_button = QPushButton('Sample Video 2: lift    ')
        self.video2_button.clicked.connect(self.selectVideo2)
        self.video3_button = QPushButton('Sample Video 3: building')
        self.video3_button.clicked.connect(self.selectVideo3)

        self.select_button = QPushButton('Select Your Own Video')
        self.select_button.clicked.connect(self.selectVideo)

        self.process_button = QPushButton('Process Video')
        self.process_button.clicked.connect(self.processVideo)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.video1_button)
        button_layout.addWidget(self.video2_button)
        button_layout.addWidget(self.video3_button)
        button_layout.addWidget(self.select_button)
        button_layout.addStretch(1)

        process_button_layout = QVBoxLayout()
        process_button_layout.addStretch(1)
        process_button_layout.addWidget(self.process_button)

        main_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addLayout(process_button_layout)

        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(self.file_path_label)
        bottom_layout.addLayout(main_layout)
        bottom_layout.addWidget(self.image_label)

        self.setLayout(bottom_layout)

    def selectVideo(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4)",
                                                   options=options)
        if file_path:
            self.file_path_label.setText("Selected Video File:\n" + file_path)
            self.video_path = file_path

    def selectVideo1(self):
        self.video_path = '../videos/sample_video_1.mp4'
        self.file_path_label.setText('Selected Video File:\nSample Video 1')

    def selectVideo2(self):
        self.video_path = '../videos/sample_video_2.mp4'
        self.file_path_label.setText('Selected Video File:\nSample Video 2')

    def selectVideo3(self):
        self.video_path = '../videos/sample_video_3.mp4'
        self.file_path_label.setText('Selected Video File:\nSample Video 3')

    def processVideo(self):
        if hasattr(self, 'video_path'):
            file_name = self.video_path.split('/')[-1].split('.')[0]

            # load images extracted from the video
            images = extract_frames(file_name, self.video_path)
            print("[INFO] loading images...")

            # Split the list of image paths into two halves
            n = len(images) // 2
            print(f'[TEST] num_of_frames: {2 * n}')
            right_images = images[n:]
            left_images = images[:n]
            # right_images = right_images[::-1]

            # Stitch images in the left half
            current_left_img = left_images[0]
            for img in left_images[1:]:
                print("[INFO] Stitching {} and {}...".format(current_left_img, img))
                current_left_img = stitch(current_left_img, img)

            # Stitch images in the right half
            current_right_img = right_images[0]
            for img in right_images[1:]:
                print("[INFO] Stitching {} and {}...".format(current_right_img, img))
                current_right_img = stitch(current_right_img, img)

            current_right_img_crop = crop_image(current_right_img)
            current_left_img_crop = crop_image(current_left_img)
            cv2.imwrite('../results/' + file_name + '_right.jpg', current_right_img)
            cv2.imwrite('../results/' + file_name + '_left.jpg', current_left_img)
            cv2.imwrite('../results/' + file_name + '_right_crop.jpg', current_right_img_crop)
            cv2.imwrite('../results/' + file_name + '_left_crop.jpg', current_left_img_crop)

            # Stitch the left and right panoramas
            print("[INFO] Join left and right parts...")
            result = stitch(current_right_img, current_left_img)
            cv2.imwrite('../results/' + file_name + '.jpg', result)
            result_crop = crop_image(result)
            cv2.imwrite('../results/' + file_name + '_crop.jpg', result_crop)

            # Convert result image to pixmap and display in label
            # result_pixmap = QPixmap('../results/' + file_name + '_crop.jpg')
            result_pixmap = QPixmap('../results/' + file_name + '_crop.jpg')
            scaled_pixmap = result_pixmap.scaledToWidth(1000)
            self.image_label.setPixmap(scaled_pixmap)
            print(f"[INFO] {file_name} panorama complete!")
        else:
            print("Please select a video file first.")


def crop_image(img):
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


# def crop_image(img):
#     """
#     Complete the image by removing any black edges and irregular shapes.
#     :param img: image with black edges and irregular shape
#     :return: cropped image after completion
#     Reference: https://pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
#     """
#     # extract the contour of the image
#     img_border = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
#     img_gray = cv2.cvtColor(img_border, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale
#     _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)  # threshold the image
#     _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # compute the bounding rectangle with the minimum contour
#     mask = np.zeros(thresh.shape, dtype="uint8")  # allocate memory
#     x, y, w, h = cv2.boundingRect(contours[0])
#     cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
#
#     # create copies of the mask
#     min_rect = mask.copy()  # minimum rectangular region
#     sub = mask.copy()  # determine whether keep reducing the size of min_rect, i.e. a counter
#
#     # operate erosion to reduce the size of min_rect
#     while cv2.countNonZero(sub) > 0:
#         # loop until the pixels of min_rect are all zeros
#         min_rect = cv2.erode(min_rect, None)
#         sub = cv2.subtract(min_rect, thresh)
#
#     # use min_rect to crop the image
#     _, contours, _ = cv2.findContours(min_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:  # Check if contours were found
#         x, y, w, h = cv2.boundingRect(contours[0])
#         img_crop = img[y:y + h, x:x + w]
#     else:
#         # Handle the case where no contours were found, perhaps by returning the original image
#         img_crop = img.copy()  # Return a copy of the original image
#
#     return img_crop

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoStitcherApp()
    window.show()
    sys.exit(app.exec_())
