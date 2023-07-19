import glob

import cv2
import numpy as np
import os
import shutil

from detect import detect
from utils_LP import character_recog_CNN, crop_img, crop_n_rotate_LP

def process_image(image_path, model_LP, model_char, device):

    source_img = cv2.imread(image_path)
    # cv2.imwrite(os.path.join("test", "1source_img.jpg"), source_img)
    image_name = os.path.basename(image_path)
    print('########################################################################')
    print(image_name)

    #################################################### Detection ####################################################

    # print('Detecting motorcycle...')
    pred_motorcycle, motorcycle_detected_img = detect(model_LP, source_img, device, imgsz=640, classes=1)
    # cv2.imwrite(os.path.join("test", "2motorcycle_detected_img.jpg"), motorcycle_detected_img)
    if pred_motorcycle is None:
        print('No motorcycle detected.')
    else:
        # Iterate over predicted motorcycle bounding boxes
        for *xyxy, conf, cls in reversed(pred_motorcycle):
            # Crop motorcycle
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            motorcycle_cropped_img = crop_img(source_img, x1, y1, x2, y2)
            motorcycle_cropped_img_copy = motorcycle_cropped_img.copy()
            # cv2.imwrite(os.path.join("test", "3motorcycle_cropped_img.jpg"), motorcycle_cropped_img)

            # print('Detecting Helmet/No Helmet...')
            helmet = 'Detection of Helmet/No Helmet Failed'
            pred_head, head_detected_img = detect(model_LP, motorcycle_cropped_img, device, imgsz=640, classes=2)
            if pred_head is None:
                pred_head, head_detected_img = detect(model_LP, motorcycle_cropped_img, device, imgsz=640, classes=3)
                # cv2.imwrite(os.path.join("test", "4head_detected_img.jpg"), head_detected_img)

                if pred_head is not None:
                    helmet = 'No Helmet Detected'
            else:
                # cv2.imwrite(os.path.join("test", "5head_detected_img.jpg"), head_detected_img)
                helmet = 'Helmet Detected'

            # print(helmet)

            # print('Detecting LP...')
            pred, LP_detected_img = detect(model_LP, head_detected_img, device, imgsz=640, classes=0)
            # cv2.imwrite(os.path.join("test", "6LP_detected_img.jpg"), LP_detected_img)
            if pred is None:
                print('No LP detected.')
            else:
                detected_img = cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5) # Resize to half its size
                # cv2.imwrite(os.path.join(output_folder_path, "detected_img.jpg"), detected_img)

                c = 0
                lplist = []
                # Iterate over the predicted license plate bounding boxes
                for *xyxy, conf, cls in reversed(pred):

                    ################ Preprocessing for Hough Transform and Rotation using Hough Lines ################

                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    angle, LP_rotated = crop_n_rotate_LP(motorcycle_cropped_img_copy, x1, y1, x2, y2)
                    # cv2.imwrite(os.path.join("test", "9LP_rotated.jpg"), LP_rotated)

                    if (LP_rotated is None):  # If the rotation fails
                        continue

                    ########################################## Preprocessing ##########################################

                    LP_rotated_copy_for_segmentation = LP_rotated.copy()
                    gray_image = cv2.cvtColor(LP_rotated_copy_for_segmentation, cv2.COLOR_BGR2GRAY)

                    _, binary = cv2.threshold(gray_image, 255, 255, cv2.THRESH_OTSU)  # Thresholding to get a binary img
                    # cv2.imwrite(os.path.join("test", "8binary.jpg"), binary)

                    ################################### Segmentation using Contours ###################################

                    LP_rotated_copy = LP_rotated.copy()
                    cont, hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # Contours = boundaries
                    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:30]  # Contours sorted based on area and limited to the top 17 contours.

                    cv2.drawContours(LP_rotated_copy, cont, -1, (100, 255, 255), 1)  # Draw (till -1 i.e all) contours
                    # cv2.imwrite(os.path.join("test", "10LP_rotated_copy_contours.jpg"), LP_rotated_copy)

                    ################################## Filter out possible characters ##################################

                    char_x = []
                    height, width, _ = LP_rotated_copy.shape
                    roiarea = height * width  #LP area

                    # Iterate over each contour
                    for ind, cnt in enumerate(cont):
                        (x, y, w, h) = cv2.boundingRect(cont[ind])
                        ratiochar = w / h
                        char_area = w * h

                        if h < 0.175 * height: # Filter out nails
                            continue
                        # char_area between (0.1/10th of LParea and 1/10th of LParea)
                        # And ratiochar(h,w) between (5x,x) and (x,1.75x)
                        elif (0.01 * roiarea < char_area < 0.1 * roiarea) and (0.2 < ratiochar <= 2):
                            char_x.append([x, y, w, h])

                    if not char_x:
                        continue

                    char_x = np.array(char_x)

                    #################### Determine Character order in Single Line / Double Line LP ####################

                    threshold_12line = char_x[:, 1].min() + (char_x[:, 3].mean() / 2)  # Line between lines in the double line LP
                    char_x = sorted(char_x, key=lambda x: x[0], reverse=False)  # Sorts the contours based on their x-coordinates i.e sort in-lne order

                    char_first_line = []
                    char_second_line = []

                    # Enters the characters in proper order and line of double line
                    for i, char in enumerate(char_x):
                        x, y, w, h = char
                        if y < threshold_12line:
                            char_first_line.append([x, y, w, h])
                        else:
                            char_second_line.append([x, y, w, h])

                    char_first_line = np.array(char_first_line)
                    char_second_line = np.array(char_second_line)

                    #################################### Filter out inner contours ####################################

                    char_double_line = []
                    for char_list in [char_first_line, char_second_line]:
                        char_double_line.append(char_list[0])
                        for i in range(len(char_list) - 1):
                            current_char = char_list[i]
                            next_char = char_list[i+1]
                            x1, y1, w1, h1 = current_char
                            x2, y2, w2, h2 = next_char
                            if x2 >= (x1 + 0.75 * w1):
                                char_double_line.append([x2, y2, w2, h2])

                    char_double_line = np.array(char_double_line)

                    ###################################### Character recognition ######################################

                    strFinalString = ""
                    for i, char in enumerate(char_double_line):
                        x, y, w, h = char
                        cv2.rectangle(LP_rotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        imgROI = binary[y:y + h, x:x + w]
                        text = character_recog_CNN(model_char, imgROI)
                        strFinalString += text

                    cv2.putText(LP_detected_img, strFinalString, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 2,
                                (255, 255, 0),
                                2)

                    # cv2.imwrite(os.path.join("test", "14LP_rotated_segmented.jpg"), LP_rotated)
                    # cv2.imwrite(os.path.join(output_folder_path, "segmented_img.jpg"), LP_rotated)
                    print('License Plate_{}:'.format(c), strFinalString)
                    lplist.append(strFinalString)
                    c += 1

                # cv2.imwrite(os.path.join("test", "15LP_detected_img.jpg"), LP_detected_img)
                final_result = cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5)
                # cv2.imwrite(os.path.join(output_folder_path, "final_result.jpg"), final_result)

                # print('Finally Done!')

                return detected_img, lplist[0]

import torch
from models.experimental import attempt_load
from src.char_classification.model import CNN_Model

def main():

    CHAR_CLASSIFICATION_WEIGHTS = 'test_data/cnn_weights.h5'
    LP_weights = 'test_data/yolov7_weights_1000imgs_4classes_50epoch.pt'

    model_char = CNN_Model(trainable=False).model
    model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
    device = torch.device('cpu')
    model_LP = attempt_load(LP_weights, map_location=device)

    test_folder_path = "test"  # path where the processed images will be stored
    if os.path.exists(test_folder_path):
        shutil.rmtree(test_folder_path)
    os.makedirs(test_folder_path)

    image_folder_path = 'C:/Users/annal/PycharmProjects/BE-Project_Cpu - Copy/Dataset/perfect/*.jpg'
    image_paths = glob.glob(image_folder_path)

    for path in image_paths:
        detected_image, license_plate = process_image(path, model_LP, model_char, device)
        image_name = os.path.basename(path)
        cv2.imwrite(os.path.join(test_folder_path, image_name), detected_image)

if __name__ == "__main__":
    main()