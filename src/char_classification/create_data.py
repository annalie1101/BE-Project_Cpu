import os
import numpy as np
import cv2

path = "./data/categorized/digits/"  # Directory with categorized digit image folders
data = []  # List for image-label pairs


# Iterate through each directory(Categorized digit image folder) in path and create appropriate label
for fi in os.listdir(path):
    if fi == "0":
        label = 24
    elif fi == "1":
        label = 25
    elif fi == "2":
        label = 26
    elif fi == "3":
        label = 27
    elif fi == "4":
        label = 28
    elif fi == "5":
        label = 29
    elif fi == "6":
        label = 30
    elif fi == "7":
        label = 31
    elif fi == "8":
        label = 32
    elif fi == "9":
        label = 33
    else:
        label = -1
        ValueError("Don't match file")

    img_fi_path = os.listdir(path + fi)  # List of images paths in each digit category
    for img_path in img_fi_path:
        img = cv2.imread(path + fi + "/" + img_path, cv2.IMREAD_GRAYSCALE)  # Loaded using cv2.imread() and converted to grayscale (cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), cv2.INTER_AREA)  # Resized to a shape of (28, 28)
        img = img.reshape((28, 28, 1))  # Re-shaped to (28, 28, 1) by adding a new axis(channel dimension)
        data.append((img, label))


np.save("./data/digits.npy", data)


path = "./data/categorized/alphas/"
data = []


for fi in os.listdir(path):
    if fi == "A":
        label = 0
    elif fi == "B":
        label = 1
    elif fi == "C":
        label = 2
    elif fi == "D":
        label = 3
    elif fi == "E":
        label = 4
    elif fi == "F":
        label = 5
    elif fi == "G":
        label = 6
    elif fi == "H":
        label = 7
    elif fi == "J":
        label = 8
    elif fi == "K":
        label = 9
    elif fi == "L":
        label = 10
    elif fi == "M":
        label = 11
    elif fi == "N":
        label = 12
    elif fi == "P":
        label = 13
    elif fi == "Q":
        label = 14
    elif fi == "R":
        label = 15
    elif fi == "S":
        label = 16
    elif fi == "T":
        label = 17
    elif fi == "U":
        label = 18
    elif fi == "V":
        label = 19
    elif fi == "W":
        label = 20
    elif fi == "X":
        label = 21
    elif fi == "Y":
        label = 22
    elif fi == "Z":
        label = 23
    else:
        label = -1
        ValueError("Don't match file")

    img_fi_path = os.listdir(path + fi)
    for img_path in img_fi_path:
        img = cv2.imread(path + fi + "/" + img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
        img = img.reshape((28, 28, 1))
        data.append((img, label))


np.save("./data/alphas.npy", data)

