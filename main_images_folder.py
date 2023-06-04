import cv2
import numpy as np
import torch
import os

from detect import detect
from models.experimental import attempt_load
from src.char_classification.model import CNN_Model
from utils_LP import character_recog_CNN, crop_img, crop_n_rotate_LP

import glob
from IPython.display import Image, display

Min_char = 0.01
Max_char = 0.09
#image_path = 'test_data/full_test_img.jpg'
CHAR_CLASSIFICATION_WEIGHTS = 'test_data/mrzaizai_cnn.h5'
LP_weights = 'test_data/yolov7_weights_1000imgs_4classes_50epoch.pt'

model_char = CNN_Model(trainable=False).model
model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)

device = torch.device('cpu')

model_LP = attempt_load(LP_weights, map_location=device)

image_folder_path = 'Motorcycle-LP-9/test/images/*.jpg'
image_paths = glob.glob(image_folder_path)

for path in image_paths:
    source_img = cv2.imread(path)
    #cv2.imshow('input', cv2.resize(source_img, dsize=None, fx=0.5, fy=0.5))
    image_name = os.path.basename(path)
    print('########################################################################')
    print(image_name)

    # Detect motorcycle
    print('Detecting motorcycle...')
    pred_motorcycle, motorcycle_detected_img = detect(model_LP, source_img, device, imgsz=640, classes=1)
    if pred_motorcycle is None:
        print('No motorcycle detected.')
    else:
        #cv2.imshow('motorcycle_detected_img', cv2.resize(motorcycle_detected_img, dsize=None, fx=0.5, fy=0.5))
        for *xyxy, conf, cls in reversed(pred_motorcycle):
            # Crop motorcycle
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            motorcycle_cropped_img = crop_img(source_img, x1, y1, x2, y2)
            #cv2.imshow('motorcycle_cropped_img', cv2.resize(motorcycle_cropped_img, dsize=None, fx=0.5, fy=0.5))

            # Detect helmet or non_helmet
            print('Detecting Helmet/No Helmet...')
            helmet = 'Detection of Helmet/No Helmet Failed'
            pred_head, head_detected_img = detect(model_LP, motorcycle_cropped_img, device, imgsz=640, classes=2)
            if pred_head is None:
                pred_head, head_detected_img = detect(model_LP, motorcycle_cropped_img, device, imgsz=640, classes=3)
                if pred_head is not None:
                    helmet = 'No Helmet Detected'
            else:
                helmet = 'Helmet Detected'
            print(helmet)
            #cv2.imshow('head_detected_img', cv2.resize(head_detected_img, dsize=None, fx=0.5, fy=0.5))

            # Detect LP
            print('Detecting LP...')
            pred, LP_detected_img = detect(model_LP, motorcycle_cropped_img, device, imgsz=640, classes=0)
            if pred is None:
                print('No LP detected.')
            else:
                #cv2.imshow('LP_detected_img', cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5))

                c = 0
                for *xyxy, conf, cls in reversed(pred):
                    # Crop and Rotate LP
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(motorcycle_cropped_img, x1, y1, x2, y2)
                    if (rotate_thresh is None) or (LP_rotated is None):
                        continue
                    #cv2.imshow('LP_rotated', LP_rotated)
                    #cv2.imshow('rotate_thresh', rotate_thresh)

                    #################### Prepocessing and Character segmentation ####################
                    LP_rotated_copy = LP_rotated.copy()
                    cont, hier = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:17]

                    #cv2.imshow('rotate_thresh', rotate_thresh)
                    cv2.drawContours(LP_rotated_copy, cont, -1, (100, 255, 255),
                                     2)  # Draw contours of characters in a LP
                    # cv2.imshow('rotate_img',rotate_img)

                    ##################### Filter out characters #################
                    char_x_ind = {}
                    char_x = []
                    height, width, _ = LP_rotated_copy.shape
                    roiarea = height * width

                    for ind, cnt in enumerate(cont):
                        (x, y, w, h) = cv2.boundingRect(cont[ind])
                        ratiochar = w / h
                        char_area = w * h
                        # cv2.putText(LP_rotated_copy, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
                        # cv2.putText(LP_rotated_copy, str(ratiochar), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
                        if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                            char_x.append([x, y, w, h])

                    if not char_x:
                        continue

                    char_x = np.array(char_x)
                    #cv2.imshow('LP_rotated_copy', LP_rotated_copy)

                    ############ Character recognition ##########################

                    threshold_12line = char_x[:, 1].min() + (char_x[:, 3].mean() / 2)
                    char_x = sorted(char_x, key=lambda x: x[0], reverse=False)
                    strFinalString = ""
                    first_line = ""
                    second_line = ""

                    skip = False
                    for i, char in enumerate(char_x):
                        if skip:
                            skip = False
                            continue
                        x, y, w, h = char
                        cv2.rectangle(LP_rotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        imgROI = rotate_thresh[y:y + h, x:x + w]
                        #cv2.imshow('imgROI', imgROI)
                        text = character_recog_CNN(model_char, imgROI)

                        if text == '0':
                            skip = True

                        if text == 'Background':
                            text = ''

                        if y < threshold_12line:
                            first_line += text
                        else:
                            second_line += text

                    strFinalString = first_line + second_line
                    cv2.putText(LP_detected_img, strFinalString, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 2,
                                (255, 255, 0),
                                2)
                    #cv2.imshow('charac', LP_rotated_copy)
                    #cv2.imshow('LP_rotated_{}'.format(c), LP_rotated)
                    cv2.imshow(image_name, LP_rotated)
                    print('License Plate_{}:'.format(c), strFinalString)
                    c += 1

                #cv2.imshow('final_result', cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5))
                print('Finally Done!')
                #cv2.waitKey(0)


