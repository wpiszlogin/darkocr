import os
import operator

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

# data settings
pickle_path = 'train_data/train.pkl'
classes_count = 36
image_dim = 56
test_data_ratio = 0.2
# note: N* is a one-piece collection, it's probably data set error
decoding_list = [
    '6', 'P', '0', 'V', 'W', '3', 'A', '8', 'T', '1',
    'O', '9', 'H', 'R', 'N', '7', 'K', 'L', 'G', '4',
    'Y', 'C', 'E', 'J', '5', 'I', 'S', '2', 'F', 'Z',
    'N*', 'Q', 'M', 'B', 'D', 'U']


class ImageData:
    def __init__(self):
        self.data_x = np.array([])
        self.data_y = np.array([])

    @staticmethod
    def encode(char_s):
        return decoding_list.index(char_s.upper())

    @staticmethod
    def decode(char_i, do_decode=True):
        if do_decode:
            return decoding_list[char_i]
        else:
            return char_i

    def read_data(self, path):
        with open(path, "rb") as pickle_data:
            self.data_x, self.data_y = pickle.load(pickle_data)

    def read_reshaped_data(self, path):
        self.read_data(path)
        self.data_x = self.data_x.reshape(self.data_x.shape[0], image_dim, image_dim)
        self.data_y = self.data_y.reshape(self.data_y.shape[0])

    @staticmethod
    def save_array_to_png(a, path):
        im = Image.fromarray(a*255)
        im.save(path)

    def save_data_set_to_png(self, path='train_data/png/'):
        for i in range(len(self.data_x)):
            final_path = path + '/' + str(self.data_y[i]) + '/'
            if not os.path.exists(final_path):
                os.makedirs(final_path)

            self.save_array_to_png(self.data_x[i], final_path + str(i) + '#' + str(self.data_y[i]) + '.png')

    def read_augmented_data_set(self, path='train_data/png/', classes=[0, 1, 2]):
        path = 'train_data/read_test/0/'
        path += '*.png'

        image_list = []
        for filename in glob.glob(path):
            print(filename)
            im = Image.open(filename)
            image_list.append(np.array(im, dtype='d')/255)

    def find_random_indexes(self, char_i, count):
        # find indexes where label == char
        char_args = np.argwhere(self.data_y == char_i)
        char_args = char_args.reshape(char_args.shape[0])
        # select random elements
        if char_args.size > 0:
            return np.random.choice(char_args, count)
        else:
            return np.empty(0)

    def calc_pixels_mean(self):
        labels_mean = np.zeros(classes_count)
        for i in range(len(self.data_x)):
            labels_mean[self.data_y[i]] += np.sum(self.data_x[i])

        unique, counts = np.unique(self.data_y, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        for k, v in counts_dict.items():
            labels_mean[k] /= v

        return labels_mean

    def perpare_data_set_for_training(self):
        print('this is in progress')
        # # building final training data
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

    # visual functions
    def show_all_chars(self, do_decode=True, examples_count=10):
        col = 0
        plt.figure(figsize=(20, 9))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        for char_i in range(classes_count):
            char_indexes = self.find_random_indexes(char_i, examples_count)
            # show them
            row = 0
            for label_index in char_indexes:
                # goes through columns
                plt.subplot(examples_count, classes_count, row * classes_count + col + 1)
                # we need to reshape to image_dim x image_dim
                plt.imshow(self.data_x[label_index], cmap=plt.cm.binary)
                plt.xlabel(self.decode(self.data_y[label_index], do_decode))
                plt.xticks([])
                plt.yticks([])
                row += 1
            col += 1
        fig = plt.gcf()
        fig.canvas.set_window_title('All characters')
        plt.show()

    def show_chars_data(self, char_s=None, do_decode=True, cols_count=27, rows_count=10):
        char = None
        if char_s is not None:
            char = self.encode(char_s)

        char_indexes = []
        if char is not None:
            char_indexes = self.find_random_indexes(char, cols_count * rows_count)

        plt.figure(figsize=(20, 11))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        for i in range(cols_count * rows_count):
            if i < len(self.data_x) and i < len(self.data_y):
                index = i
                if i < len(char_indexes):
                    index = char_indexes[i]

                plt.subplot(rows_count, cols_count, i + 1)
                plt.imshow(self.data_x[index], cmap=plt.cm.binary)
                plt.xlabel(self.decode(self.data_y[index], do_decode))
                plt.xticks([])
                plt.yticks([])
        fig = plt.gcf()
        if char is not None:
            fig.canvas.set_window_title('Random data of label={} ({})'.format(char, str(self.decode(char))))
        else:
            fig.canvas.set_window_title('Continuous data set from beginning')
        plt.show()

    def show_data_histogram(self):
        plt.figure(figsize=(12, 4))
        plt.hist(self.data_y, bins=range(classes_count + 1), edgecolor='black', facecolor='blue', align='left')
        plt.xticks(range(classes_count))
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title('Histogram of data set')
        fig = plt.gcf()
        fig.canvas.set_window_title('Histogram of data set')
        plt.show()

    def print_labels_count(self, sort=False):
        unique, counts = np.unique(self.data_y, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        if sort:
            sorted_counts = sorted(counts_dict.items(), key=operator.itemgetter(1))
            for k, v in sorted_counts:
                print('{} ({}) = {}'.format(k, self.decode(k), v))
        else:
            for k, v in counts_dict.items():
                print('{} ({}) = {}'.format(k, self.decode(k), v))
