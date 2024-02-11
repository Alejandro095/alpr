from ultralytics import YOLO
import cv2

import numpy as np


from util import read_license_plate

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample-4.mp4')

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            # process license plate
            gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(threshold)

            print(f'License plate text: {license_plate_text}')

































# # read frames
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         # detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate

#             # crop license plate
#             license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

#             cv2.imshow('License plate', license_plate_crop)
#             cv2.waitKey(0)

#             # process license plate
#             thresholded = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

#             cv2.imshow('License plate', license_plate_crop_processed)
#             cv2.waitKey(0)

#             license_plate_crop_processed = cv2.threshold(license_plate_crop_processed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#             cv2.imshow('License plate', license_plate_crop_processed)
#             cv2.waitKey(0)

#             license_plate_crop_processed = cv2.distanceTransform( , cv2.DIST_L2, 5)
#             license_plate_crop_processed = cv2.normalize(license_plate_crop_processed, license_plate_crop_processed, 0, 1.0, cv2.NORM_MINMAX)
#             license_plate_crop_processed = (license_plate_crop_processed * 255).astype(np.uint8)

#             cv2.imshow('License plate', license_plate_crop_processed)
#             cv2.waitKey(0)

#             license_plate_crop_processed = cv2.threshold(license_plate_crop_processed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#             cv2.imshow('License plate', license_plate_crop_processed)
#             cv2.waitKey(0)

#             # read license plate number
#             license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_processed)

#             print(f'License plate text: {license_plate_text}')

#             cv2.imshow('License plate', license_plate_crop_processed)

#             cv2.waitKey(0)