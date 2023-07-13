import cv2
import numpy as np

import yaml

from Preprocess import preprocess, Hough_transform, rotation_angle, rotate_LP

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'J', 9: 'K', 10: 'L', 11: 'M', 12: 'N',
              13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'U', 19: 'V', 20: 'W', 21: 'X', 22: 'Y', 23: 'Z',
              24: '0', 25: '1', 26: '2', 27: '3', 28: '4', 29: '5', 30: '6',  31: '7', 32: '8', 33: '9'}


def create_yaml():
    """
    Creates a YAML file named data.yaml
    """
    data_yaml = dict(
        train='../input/vietnamese-license-plate/train',
        val='../input/vietnamese-license-plate/valid',
        nc=1,
        names=['License Plate']
    )

    with open('data.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)


def character_recog_CNN(model, img, dict=ALPHA_DICT):
    '''
    Performs OCR using the CNN model i.e. Turn character image to text
    Resize, Reshape and Expand dimensions to match the input shape expected by the model
    Predict result and return in text format using ALPHA_DICT

    :param model: Model character recognition
    :param img: threshold image no fixed size (white character, black background)
    :param dict: alphabet dictionary
    :return: ASCII text
    '''

    # Resize, Reshape and Expand dimensions to match the input shape expected by the model
    imgROI = cv2.resize(img, (28, 28), cv2.INTER_AREA)
    imgROI = imgROI.reshape((28, 28, 1))
    imgROI = np.array(imgROI)
    imgROI = np.expand_dims(imgROI, axis=0)

    result = model.predict(imgROI, verbose='2')
    result_idx = np.argmax(result, axis=1) # index of the maximum value in the result
    return ALPHA_DICT[result_idx[0]]

def crop_img(source_img, x1, y1, x2, y2):
    '''
    Crop detected object from original image based on coords

    :param source_img:
    :param x1,y1,x2,y2: coordinates of detected objects
    :return: cropped_img
    '''

    w = int(x2 - x1)
    h = int(y2 - y1)
    cropped_img = source_img[y1:y1 + h, x1:x1 + w]
    cropped_img_copy = cropped_img.copy()  # Create copy of the cropped image to avoid modifying the original image.
    return cropped_img_copy

def crop_n_rotate_LP(source_img, x1, y1, x2, y2):
    '''
    Crop and rotate License Plate from original image based on coords
    Calculates w, h and ratio of LP, Crops LP, applies preprocess(), canny edge, dilate
    Applies Hough_Transform() to get lines of image
    Calculates the rotation angle using rotation_angle()
    Rotates the cropped LP using rotate_LP
    Returns rotation angle, rotated thresholded image, and rotated LP image.

    :param source_img:
    :param x1,y1,x2,y2: coordinates of License Plate
    :return: angle, rotate_thresh, LP_rotated
    '''
    w = int(x2 - x1)
    h = int(y2 - y1)
    ratio = w / h
    print ('ratio',ratio)
    #if 0.8 <= ratio <= 1.5 or 3.5 <= ratio <= 6.5:
    #if 0.8 <= ratio <= 2 or 3.5 <= ratio <= 6.5:
    cropped_LP = source_img[y1:y1 + h, x1:x1 + w]
    cropped_LP_copy = cropped_LP.copy()
    # cv2.imwrite('doc/cropped_LP2.png',cropped_LP)

    imgGrayscaleplate, imgThreshplate = preprocess(cropped_LP)
    canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=2)

    linesP = Hough_transform(dilated_image, nol=6)
    for i in range(0, len(linesP)):
        l = linesP[i][0].astype(int)
        # cv2.line(cropped_LP_copy, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    angle = rotation_angle(linesP)
    rotate_thresh = rotate_LP(imgThreshplate, angle)
    LP_rotated = rotate_LP(cropped_LP, angle)
    #else:
    #    angle, rotate_thresh, LP_rotated = None, None, None

    return angle, rotate_thresh, LP_rotated


def main():
    # create_yaml()
    print('haha')


if __name__ == "__main__":
    main()

