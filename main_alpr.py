import cv2
import numpy as np
import os
import shutil

from detect import detect
from utils_LP import character_recog_CNN, crop_img, crop_n_rotate_LP

def process_image(image_path, model_LP, model_char, device):
    output_folder_path = "static"  # Update with your desired folder path

    Min_char = 0.01
    Max_char = 0.09

    source_img = cv2.imread(image_path)
    original_img = cv2.resize(source_img, dsize=None, fx=0.5, fy=0.5)
    cv2.imwrite(os.path.join(output_folder_path, "original_img.jpg"), original_img)
    image_name = os.path.basename(image_path)
    print('########################################################################')
    print(image_name)

    # Detect motorcycle
    print('Detecting motorcycle...')
    pred_motorcycle, motorcycle_detected_img = detect(model_LP, source_img, device, imgsz=640, classes=1)
    if pred_motorcycle is None:
        print('No motorcycle detected.')
    else:
        for *xyxy, conf, cls in reversed(pred_motorcycle):
            # Crop motorcycle
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            motorcycle_cropped_img = crop_img(source_img, x1, y1, x2, y2)

            # Detect helmet or non_helmet
            print('Detecting Helmet/No Helmet...')
            helmet = 'Detection of Helmet/No Helmet Failed'
            pred_head, head_detected_img = detect(model_LP, motorcycle_cropped_img, device, imgsz=640, classes=2)
            if pred_head is None:
                pred_head, head_detected_img = detect(model_LP, motorcycle_cropped_img, device, imgsz=640, classes=3)

                if pred_head is not None:
                    helmet = 'No Helmet Detected'
                    shutil.copyfile('test_data/nohelmet.jpg', 'static/alert.jpg')
            else:
                helmet = 'Helmet Detected'
                shutil.copyfile('test_data/helmet.jpg', 'static/alert.jpg')

            print(helmet)

            # Detect LP
            print('Detecting LP...')
            pred, LP_detected_img = detect(model_LP, head_detected_img, device, imgsz=640, classes=0)
            if pred is None:
                print('No LP detected.')
            else:
                detected_img = cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5)
                cv2.imwrite(os.path.join(output_folder_path, "detected_img.jpg"), detected_img)

                c = 0
                lplist = []
                for *xyxy, conf, cls in reversed(pred):
                    # Crop and Rotate LP
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(motorcycle_cropped_img, x1, y1, x2, y2)
                    if (rotate_thresh is None) or (LP_rotated is None):
                        continue

                    #################### Prepocessing and Character segmentation ####################
                    LP_rotated_copy = LP_rotated.copy()
                    cont, hier = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:17]

                    cv2.drawContours(LP_rotated_copy, cont, -1, (100, 255, 255), 2)  # Draw contours of characters in a LP

                    ##################### Filter out characters #################
                    char_x_ind = {}
                    char_x = []
                    height, width, _ = LP_rotated_copy.shape
                    roiarea = height * width

                    for ind, cnt in enumerate(cont):
                        (x, y, w, h) = cv2.boundingRect(cont[ind])
                        ratiochar = w / h
                        char_area = w * h

                        if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                            char_x.append([x, y, w, h])

                    if not char_x:
                        continue

                    char_x = np.array(char_x)

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

                    cv2.imwrite(os.path.join(output_folder_path, "segmented_img.jpg"), LP_rotated)
                    print('License Plate_{}:'.format(c), strFinalString)
                    lplist.append(strFinalString)
                    c += 1

                final_result = cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5)
                cv2.imwrite(os.path.join(output_folder_path, "final_result.jpg"), final_result)

                print('Finally Done!')

                return lplist[0]