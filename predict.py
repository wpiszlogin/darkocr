import operator
import random

import pickle
from PIL import Image, ImageFilter
import Augmentor
from Augmentor.Operations import Operation
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Flatten, Dropout, Dense

classes_count = 36
image_dim = 56
test_data_ratio = 0.2
# note: N* is a one-piece collection, it's probably data set error
decoding_list = [
    '6', 'P', '0', 'V', 'W', '3', 'A', '8', 'T', '1',
    'O', '9', 'H', 'R', 'N', '7', 'K', 'L', 'G', '4',
    'Y', 'C', 'E', 'J', '5', 'I', 'S', '2', 'F', 'Z',
    'N*', 'Q', 'M', 'B', 'D', 'U']


def predict(input_data):
    output_data = 'done'
    return output_data


def read_data(path):
    with open(path, "rb") as pickle_data:
        train_data = pickle.load(pickle_data)
        pickle_data
        return train_data


def read_reshaped_data(path):
    data_raw_x, data_raw_y = read_data(path)
    data_x = data_raw_x.reshape(data_raw_x.shape[0], image_dim, image_dim)
    data_y = data_raw_y.reshape(data_raw_y.shape[0])
    return data_x, data_y


def save_array_to_png(a, path):
    im = Image.fromarray(a*255)
    im.save(path)


def save_arrays_to_png(data_x, data_y, path='train_data/png/'):
    for i in range(len(data_x)):
        save_array_to_png(data_x[i], path+str(i)+'_'+str(data_y[i])+'.png')


def encode(char):
    return decoding_list.index(char.upper())


def decode(char_i, do_decode=True):
    if do_decode:
        return decoding_list[char_i]
    else:
        return char_i


def find_random_indexes(data_y, char, count):
    # find indexes where label == char
    char_args = np.argwhere(data_y == char)
    char_args = char_args.reshape(char_args.shape[0])
    # select random elements
    if char_args.size > 0:
        return np.random.choice(char_args, count)
    else:
        return np.empty(0)


def show_all_chars(data_x, data_y, do_decode=True, examples_count=10):
    col = 0
    plt.figure(figsize=(20, 9))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for char in range(classes_count):
        char_indexes = find_random_indexes(data_y, char, examples_count)
        # show them
        row = 0
        for label_index in char_indexes:
            # goes through columns
            plt.subplot(examples_count, classes_count, row * classes_count + col + 1)
            # we need to reshape to image_dim x image_dim
            plt.imshow(data_x[label_index], cmap=plt.cm.binary)
            plt.xlabel(decode(data_y[label_index], do_decode))
            plt.xticks([])
            plt.yticks([])
            row += 1
        col += 1
    fig = plt.gcf()
    fig.canvas.set_window_title('All characters')
    plt.show()


def show_chars_data(data_x, data_y, char=None, do_decode=True, cols_count=27, rows_count=10):
    char_indexes = []
    if char is not None:
        char_indexes = find_random_indexes(data_y, char, cols_count * rows_count)

    plt.figure(figsize=(20, 11))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for i in range(cols_count * rows_count):
        if i < len(data_x) and i < len(data_y):
            index = i
            if i < len(char_indexes):
                index = char_indexes[i]

            plt.subplot(rows_count, cols_count, i + 1)
            plt.imshow(data_x[index], cmap=plt.cm.binary)
            plt.xlabel(decode(data_y[index], do_decode))
            plt.xticks([])
            plt.yticks([])
    fig = plt.gcf()
    if char is not None:
        fig.canvas.set_window_title('Random data of label={} ({})'.format(char, str(decode(char))))
    else:
        fig.canvas.set_window_title('Continuous data set from beginning')
    plt.show()


def show_data_histogram(data_y):
    plt.figure(figsize=(12, 4))
    plt.hist(data_y, bins=range(classes_count + 1), edgecolor='black', facecolor='blue', align='left')
    plt.xticks(range(classes_count))
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Histogram of data set')
    fig = plt.gcf()
    fig.canvas.set_window_title('Histogram of data set')
    plt.show()


def show_labels_count(data_y, sort=False):
    unique, counts = np.unique(data_y, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    if sort:
        sorted_counts = sorted(counts_dict.items(), key=operator.itemgetter(1))
        for k, v in sorted_counts:
            print('{} ({}) = {}'.format(k, decode(k), v))
    else:
        for k, v in counts_dict.items():
            print('{} ({}) = {}'.format(k, decode(k), v))


def calc_pixels_mean(data_x, data_y):
    labels_mean = np.zeros(classes_count)
    for i in range(len(data_x)):
        labels_mean[data_y[i]] += np.sum(data_x[i])

    unique, counts = np.unique(data_y, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    for k, v in counts_dict.items():
        labels_mean[k] /= v

    return labels_mean


# reading data
train_x, train_y = read_reshaped_data('train_data/train.pkl')
# visualization
show_all_chars(train_x, train_y)
show_chars_data(train_x, train_y)
show_chars_data(train_x, train_y, char=encode('p'))
# statistics
show_data_histogram(train_y)
show_labels_count()


# data augmentation
class ErosImage(Operation):
    def __init__(self, probability, max_erosion, max_dilation):
        Operation.__init__(self, probability)
        self.max_erosion = max_erosion
        self.max_dilation = max_dilation

    def perform_operation(self, image):
        image_mod = []
        for im in image:
            r = random.randint(-self.max_erosion, self.max_dilation)
            im_mod = im
            if r > 0:
                im_mod = im.filter(ImageFilter.MaxFilter(3))
            elif r < 0:
                im_mod = im.filter(ImageFilter.MinFilter(3))
            image_mod.append(im_mod)

        return image_mod


class SmoothImage(Operation):
    def __init__(self, probability, blur=1, threshold=128):
        Operation.__init__(self, probability)
        self.blur = blur
        self.threshold = threshold

    def perform_operation(self, image):
        image_mod = []
        for im in image:
            imbl = im.filter(ImageFilter.GaussianBlur(radius=self.blur))
            im_mod = imbl.point(lambda x: 0 if x < self.threshold else 255)
            image_mod.append(im_mod)

        return image_mod


class AutoresizeImage(Operation):
    def __init__(self, probability, max_horizontal_loss, max_vertical_loss):
        Operation.__init__(self, probability)
        self.max_horizontal_loss = max_horizontal_loss
        self.max_vertical_loss = max_vertical_loss

    def perform_operation(self, image):
        image_mod = []
        for im in image:
            # if no modifications
            im_mod = im
            # calc bounding box
            bbox = im.getbbox()
            margin_tol = 0.15
            horiz_margin_tol = margin_tol * im.width
            vert_margin_tol = margin_tol * im.height
            # comparing left, upper, right, bottom
            if bbox[0] < horiz_margin_tol and bbox[1] < vert_margin_tol and im.width - bbox[2] < horiz_margin_tol and im.height - bbox[3] < vert_margin_tol:
                im_mod = Image.new('L', (im.width, im.height))
                if random.randint(0, 10) <= 7:
                    # resize horizontally
                    scal_factor = (1 - self.max_horizontal_loss * random.random())
                    i2 = im.resize((int(scal_factor*im.width), im.height))
                else:
                    # resize vertically
                    scal_factor = (1 - self.max_vertical_loss * random.random())
                    i2 = im.resize((im.width, int(scal_factor*im.height)))

                im_mod.paste(i2, (int(im.width / 2 - i2.width / 2), int(im.height / 2 - i2.height / 2)))

            image_mod.append(im_mod)

        return image_mod

p = Augmentor.Pipeline("train_data/aug/")
p.add_operation(AutoresizeImage(probability=1, max_horizontal_loss=0.4, max_vertical_loss=0.2))
p.sample(50)

# statistics for calculations
labels_mean = calc_pixels_mean(train_x, train_y)

p = Augmentor.Pipeline("train_data/aug/")
p.add_operation(ErosImage(probability=0.5, max_erosion=1, max_dilation=1))
p.random_distortion(1, 3, 3, 4)
p.skew(1, 0.2)
p.rotate_without_crop(probability=1, max_left_rotation=5, max_right_rotation=5)
p.shear(1, 5, 5)
p.add_operation(SmoothImage(probability=1))
p.sample(50)

p = Augmentor.Pipeline("train_data/aug/")
p.random_distortion(1, 2, 2, 8)
p.add_operation(AutoresizeImage(probability=1, max_horizontal_loss=0.4, max_vertical_loss=0.2))
p.rotate_without_crop(probability=1, max_left_rotation=7, max_right_rotation=7)
p.skew(1, 0.2)
p.resize(1, 116, 116)
p.add_operation(ErosImage(probability=0.5, max_erosion=1, max_dilation=1))
p.resize(1, 56, 56)
p.add_operation(SmoothImage(probability=1, blur=1, threshold=128))
p.sample(50)

# # testing model prototype
# pi = find_random_indexes(train_y, encode('a'), 1000)
# zi = find_random_indexes(train_y, encode('e'), 1000)
# pzi = np.concatenate((pi, zi))
# np.random.shuffle(pzi)
# # copying data
# train_count = int((1.0-test_data_ratio)*len(pzi))
# test_count = len(pzi) - train_count
# final_train_y = np.zeros(train_count, dtype='int64')
# final_train_x = np.zeros((train_count, image_dim, image_dim), dtype='uint8')
# final_test_y = np.zeros(test_count, dtype='int64')
# final_test_x = np.zeros((test_count, image_dim, image_dim), dtype='uint8')
#
# i = 0
# for pz in pzi:
#     if i < len(final_train_y):
#         final_train_x[i] = train_x[pz]
#         final_train_y[i] = train_y[pz]
#     else:
#         final_test_x[train_count-i] = train_x[pz]
#         final_test_y[train_count-i] = train_y[pz]
#     i += 1
#
# # show_chars_data(final_train_x, final_train_y)
# # show_chars_data(final_test_x, final_test_y)
# # adding dimension - color depth
# x = final_train_x.reshape(final_train_x.shape[0], final_train_x.shape[1], final_train_x.shape[2], 1)
# xt = final_test_x.reshape(final_test_x.shape[0], final_test_x.shape[1], final_test_x.shape[2], 1)
#
#
# # building model
# model = keras.Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(image_dim, image_dim, 1)))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(36, activation='softmax'))
#
# model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x, final_train_y, batch_size=512, validation_data=(xt, final_test_y), epochs=5, verbose=1)
