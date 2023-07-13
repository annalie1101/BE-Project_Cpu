import math

import cv2
import numpy as np

# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)  # The larger the size, the more blurred
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9


###################################################################################################
def preprocess(imgOriginal):
    '''
    Converts RGB img to grayscale, then maximizes contrast, blurs using Gaussian Filter
    And applies adaptive thresholding to create a binary image

    :param imgOriginal: RGB image (cv2)
    :return: imgGrayscale, imgThresh
    '''
    imgGrayscale = extractValue(imgOriginal)  # Converts img to grayscale
    # imgGrayscale = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY)  # should use HSV color system
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)  # to make the number plate more prominent, easy to separate from the background
    # cv2.imwrite("imgGrayscalePlusTopHatMinusBlackHat.jpg",imgMaxContrastGrayscale)
    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    # cv2.imwrite("gauss.jpg",imgBlurred)
    # Smooth the image with a 5x5 Gaussian filter, sigma = 0

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    # Create binary image
    return imgGrayscale, imgThresh


def extractValue(imgOriginal):
    """
    Takes RGB image and converts to HSV and extracts the value channel.
    """
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)  # Convert input image to HSV color space

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    # hue, saturation, intensity value
    # Do not choose RBG color because eg red image will be mixed with other colors, so it is difficult to determine "one color"
    return imgValue


def maximizeContrast(imgGrayscale):
    """
    Enhances the contrast of a grayscale image using top-hat and black-hat morphological operations.
    Contrast-Enhanced Image = (grayscale Image + top-hat image) - black-hat image
    """
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # create kernel filter

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement,
                                 iterations=10)  # highlight bright details in dark background
    # cv2.imwrite("tophat.jpg",imgTopHat)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement,
                                   iterations=10)  # Highlight dark details in light background
    # cv2.imwrite("blackhat.jpg",imgBlackHat)

    # Add grayscale and the top-hat image to obtain an intermediate image
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    # cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat


def rotation_angle(linesP):
    '''
    Takes the lines detected using the Hough transform
    Calculates angle of each and brings within range -45 to 45 degrees
    Returns the average angle

    :param linesP: matrix of hough lines and length
    :return:
    angle: array
        matrix of angles
    '''

    angles = []
    for i in range(0, len(linesP)):
        # Calculate the angle using the line coordinates
        l = linesP[i][0].astype(int)
        p1 = (l[0], l[1])
        p2 = (l[2], l[3])
        doi = (l[1] - l[3])
        ke = abs(l[0] - l[2])
        angle = math.atan(doi / ke) * (180.0 / math.pi)
        # If abs(angle) > 45 degrees, it adjusts the angle to be within -45 to 45 degrees.
        if abs(angle) > 45:  # If they find vertical lines
            angle = (90 - abs(angle)) * angle / abs(angle)
        angles.append(angle)

    angles = list(filter(lambda x: (abs(x > 3) and abs(x < 15)), angles))
    if not angles:  # If the angles is empty, assign a default angle of 0
        angles = list([0])
    angle = np.array(angles).mean()
    return angle


def rotate_LP(img, angle):
    '''
    Rotates an image based on a given angle

    :param img:
    :param angle:
    :return: rotated image
    '''
    height, width = img.shape[:2]
    ptPlateCenter = width / 2, height / 2
    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotationMatrix, (width, height))
    return rotated_img


def Hough_transform(threshold_image, nol=6):
    '''
    Performs the Hough transform on a thresholded image to detect lines
    Returns an array of line coordinates and lengths.

    :param threshold_image:
    :param nol: number of lines have longest lenght
    :return:
    linesP: array(xyxy,line_length)
        array of coordinates and length
    '''
    h, w = threshold_image.shape[:2]
    linesP = cv2.HoughLinesP(threshold_image, 1, np.pi / 180, 50, None, 50, 10)
    dist = []
    for i in range(0, len(linesP)):
        # Calculates the lengths of the lines
        l = linesP[i][0]
        d = math.sqrt((l[0] - l[2]) ** 2 + (l[1] - l[3]) ** 2)
        if d < 0.5 * max(h, w):
            d = 0
        dist.append(d)
        # cv2.line(threshold_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    dist = np.array(dist).reshape(-1, 1, 1)
    linesP = np.concatenate([linesP, dist], axis=2)
    linesP = sorted(linesP, key=lambda x: x[0][-1], reverse=True)[:nol] # sort in descending order of length

    return linesP


def main():
    Min_char = 0.01
    Max_char = 0.09
    LP_img = cv2.imread('doc/cropped_LP2.png')
    _, thresh = preprocess(LP_img)
    linesP = Hough_transform(thresh)
    cv2.imshow('thresh', thresh)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
