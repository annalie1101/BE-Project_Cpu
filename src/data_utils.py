import numpy as np
import cv2


def get_digits_data(path):
    """
    Loads data from NumPy file(containing data about digit) specified by path param.
    The data is shuffled randomly.
    Returns the loaded data.
    """

    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)
    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train digits data: ', len(data_train))

    return data_train


def get_alphas_data(path):
    """
    Loads data from NumPy file(containing data about alpha(character)) specified by path param.
    The data is shuffled randomly.
    Returns the loaded data.
    """

    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)

    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train alphas data: ', len(data_train))

    return data_train


def get_labels(path):
    """
    Reads labels from a text file specified by path param.
    Lines of the file are read and stripped of leading or trailing whitespace.
    Returns a list of labels.
    """

    with open(path, 'r') as file:
        lines = file.readlines()

    return [line.strip() for line in lines]


def draw_labels_and_boxes(image, labels, boxes):
    """
    Takes an image, labels, and bounding box coords as input.
    Draws a rectangle around the box coords and adds labels to the image.
    The modified image is returned.
    """

    x_min = round(boxes[0])
    y_min = round(boxes[1])
    x_max = round(boxes[0] + boxes[2])
    y_max = round(boxes[1] + boxes[3])

    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
    image = cv2.putText(image, labels, (x_min - 20, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.25, color=(0, 0, 255), thickness=2)

    return image


def get_output_layers(model):
    """
    Takes a model as input.
    Retrieves the names of the output layers from the model.
    Returns a list of output layer names.
    """

    layers_name = model.getLayerNames()
    output_layers = [layers_name[i - 1] for i in model.getUnconnectedOutLayers()]

    return output_layers


def order_points(coordinates):
    """
    Takes the coords of a rectangle (x_min, y_min, width, height) as input.
    Orders the coords of the rectangle in a clockwise manner (top left, top right, bottom left, bottom right).
    The ordered coords are returned as a NumPy array.
    """

    rect = np.zeros((4, 2), dtype="float32")
    x_min, y_min, width, height = coordinates

    # top left - top right - bottom left - bottom right
    rect[0] = np.array([round(x_min), round(y_min)])
    rect[1] = np.array([round(x_min + width), round(y_min)])
    rect[2] = np.array([round(x_min), round(y_min + height)])
    rect[3] = np.array([round(x_min + width), round(y_min + height)])

    return rect


def convert2Square(image):
    """
    Resize non-square image(height != width) to square one (height == width)
    If height > width, it adds zero-padding to left and right sides.
    If width > height, it adds zero-padding to top and bottom.
    If height == width, it is returned as is.
    The squared image is returned as a NumPy array.
    """

    img_h = image.shape[0]
    img_w = image.shape[1]

    # if height > width
    if img_h > img_w:
        diff = img_h - img_w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = np.zeros(shape=(img_h, (diff//2) + 1))

        squared_image = np.concatenate((x1, image, x2), axis=1)
    elif img_w > img_h:
        diff = img_w - img_h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1

        squared_image = np.concatenate((x1, image, x2), axis=0)
    else:
        squared_image = image

    return squared_image
