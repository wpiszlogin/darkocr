import os
import operator

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

# data settings
pickle_path = 'train_data/train.pkl'
png_path = 'train_data/png/'
augmented_pickle_path = 'train_data/aug_data.pkl'
classes_count = 36
image_dim = 56
test_data_ratio = 0.2
fold_count = 5
# note: N* is a one-piece collection, it's probably data set error
decoding_list = [
    '6', 'P', '0', 'V', 'W', '3', 'A', '8', 'T', '1',
    'O', '9', 'H', 'R', 'N', '7', 'K', 'L', 'G', '4',
    'Y', 'C', 'E', 'J', '5', 'I', 'S', '2', 'F', 'Z',
    'N*', 'Q', 'M', 'B', 'D', 'U']


class ImageData:
    def __init__(self):
        # origin data set
        self.origin_x = np.array([])
        self.origin_y = np.array([])

        # augmented data set
        self.train_y = np.array([])
        self.train_x = np.array([])
        self.test_y = np.array([])
        self.test_x = np.array([])

    @staticmethod
    def encode(char_s):
        return decoding_list.index(char_s.upper())

    @staticmethod
    def decode(char_i, do_decode=True):
        if do_decode:
            return decoding_list[char_i]
        else:
            return char_i

    @staticmethod
    def read_pickle(path):
        with open(path, "rb") as pickle_data:
            return pickle.load(pickle_data)

    def read_origin_data(self, path):
        x, y = self.read_pickle(path)
        self.origin_x = x.reshape(-1, image_dim, image_dim)
        self.origin_y = y.reshape(-1)

    @staticmethod
    def save_array_to_png(a, path):
        im = Image.fromarray(a*255)
        im.save(path)

    def save_data_set_to_png(self, path=png_path):
        for i in range(len(self.origin_x)):
            final_path = path + '/' + str(self.origin_y[i]) + '/'
            if not os.path.exists(final_path):
                os.makedirs(final_path)

            self.save_array_to_png(self.origin_x[i], final_path + str(i) + '#' + str(self.origin_y[i]) + '.png')

    @staticmethod
    def path_to_file_name(path):
        sl = path.rfind("/")
        bsl = path.rfind("\\")
        return path[max(sl, bsl) + 1:]

    # read augmented png folders and build data set which is easy to manipulate
    def read_augmented_data_and_process(self, in_path=png_path, out_path=None, classes_count_int=classes_count):
        print('\nRead augmented images started...\n')

        data_set_list = []
        extension = '*.png'
        aug_fold = '/augmentation/'

        for char_i in range(classes_count_int):
            print('Processing class {}'.format(char_i))
            augm_temp_list = []
            origin_list = []
            # get list of augmented images
            for im_path in glob.glob(in_path + str(char_i) + aug_fold + extension):
                im = Image.open(im_path)
                file_name = self.path_to_file_name(im_path)
                augm_temp_list.append((file_name, np.array(im, dtype='d')/255))

            # match augmented to original images
            for im_path in glob.glob(in_path + str(char_i) + '/' + extension):
                augm_list = []
                im = Image.open(im_path)
                # find augmented images from original
                file_name = self.path_to_file_name(im_path)
                matching = [a for n, a in augm_temp_list if n.startswith(str(char_i) + '_original_' + file_name)]
                # zero element is alleyways original image
                augm_list.append(np.array(im, dtype='d')/255)
                # extend the list by all augmented images
                augm_list.extend(matching)
                # this list contains all sublist of one augmentation
                origin_list.append(augm_list)

            # add sublist of one class for whole data set
            data_set_list.append(origin_list)

        if out_path is not None:
            with open(out_path, 'wb') as file:
                pickle.dump(data_set_list, file, protocol=pickle.HIGHEST_PROTOCOL)

        print('\nRead finished\n')
        return data_set_list

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    # transform data to k-fold validation set, it prevents mix augmented images between training and testing set
    def from_processed_data_to_training_set(self, data_set=None, test_fold=4, permutation=True):
        if data_set is None:
            print("\nCan't build training set. No data set!\n")
            return

        classes_count_int = len(data_set)
        print('\nBuilding training set of {} classes, fold = {} ...\n'.format(classes_count_int, test_fold))
        train_list_y = []
        train_list_x = []
        test_list_y = []
        test_list_x = []

        fold = 0
        for label in range(classes_count_int):
            e_per_label_count = 0
            for examp in data_set[label]:
                for aug in examp:
                    if fold == test_fold:
                        test_list_x.append(aug)
                        test_list_y.append(label)
                    else:
                        train_list_x.append(aug)
                        train_list_y.append(label)

                    e_per_label_count += 1

                if fold >= fold_count - 1:
                    fold = 0
                else:
                    fold += 1

            print('Class {} has {} examples ({} was original)'.format(label, e_per_label_count, len(data_set[label])))

        # transform to np array
        train_x = np.array(train_list_x)
        train_y = np.array(train_list_y)
        test_x = np.array(test_list_x)
        test_y = np.array(test_list_y)

        # examples permutation
        if permutation:
            train_x, train_y = self.unison_shuffled_copies(train_x, train_y)

        # add dimension - color depth
        self.train_x = np.expand_dims(train_x, axis=3)
        self.train_y = train_y
        self.test_x = np.expand_dims(test_x, axis=3)
        self.test_y = test_y

        print('\nTraining set is complete\n')
        return (self.train_x, self.train_y), (self.test_x, self.test_y)

    # it can be use to validate small data set
    def show_training_set(self, cols_count=27, rows_count=10):
        # multiply by 100 to easy mark test labels
        test_y_int = self.test_y + 100
        # merge sets
        merge_x = np.concatenate((self.train_x, self.test_x), axis=0)
        merge_y = np.concatenate((self.train_y, test_y_int), axis=0)
        # decrease dimension - delete color depth
        merge_x = np.squeeze(merge_x, axis=3)

        plt.figure(figsize=(20, 11))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        for i in range(cols_count * rows_count):
            if i < len(merge_x) and i < len(merge_y):
                plt.subplot(rows_count, cols_count, i + 1)
                plt.imshow(merge_x[i], cmap=plt.cm.binary)
                plt.xlabel(merge_y[i])
                plt.xticks([])
                plt.yticks([])

        fig = plt.gcf()
        fig.canvas.set_window_title('Training set validation')
        plt.show()

    def find_random_indexes(self, char_i, count):
        # find indexes where label == char
        char_args = np.argwhere(self.origin_y == char_i)
        char_args = char_args.reshape(char_args.shape[0])
        # select random elements
        if char_args.size > 0:
            return np.random.choice(char_args, count)
        else:
            return np.empty(0)

    def calc_pixels_mean(self):
        labels_mean = np.zeros(classes_count)
        for i in range(len(self.origin_x)):
            labels_mean[self.origin_y[i]] += np.sum(self.origin_x[i])

        unique, counts = np.unique(self.origin_y, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        for k, v in counts_dict.items():
            labels_mean[k] /= v

        return labels_mean

    # visual functions
    def show_origin_all_chars(self, do_decode=True, examples_count=10):
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
                plt.imshow(self.origin_x[label_index], cmap=plt.cm.binary)
                plt.xlabel(self.decode(self.origin_y[label_index], do_decode))
                plt.xticks([])
                plt.yticks([])
                row += 1
            col += 1
        fig = plt.gcf()
        fig.canvas.set_window_title('All characters')
        plt.show()

    def show_origin_chars_data(self, char_s=None, do_decode=True, cols_count=27, rows_count=10):
        char = None
        if char_s is not None:
            char = self.encode(char_s)

        char_indexes = []
        if char is not None:
            char_indexes = self.find_random_indexes(char, cols_count * rows_count)

        plt.figure(figsize=(20, 11))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        for i in range(cols_count * rows_count):
            if i < len(self.origin_x) and i < len(self.origin_y):
                index = i
                if i < len(char_indexes):
                    index = char_indexes[i]

                plt.subplot(rows_count, cols_count, i + 1)
                plt.imshow(self.origin_x[index], cmap=plt.cm.binary)
                plt.xlabel(self.decode(self.origin_y[index], do_decode))
                plt.xticks([])
                plt.yticks([])
        fig = plt.gcf()
        if char is not None:
            fig.canvas.set_window_title('Random data of label={} ({})'.format(char, str(self.decode(char))))
        else:
            fig.canvas.set_window_title('Continuous data set from beginning')
        plt.show()

    def show_origin_data_histogram(self):
        plt.figure(figsize=(12, 4))
        plt.hist(self.origin_y, bins=range(classes_count + 1), edgecolor='black', facecolor='blue', align='left')
        plt.xticks(range(classes_count))
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title('Histogram of data set')
        fig = plt.gcf()
        fig.canvas.set_window_title('Histogram of data set')
        plt.show()

    def print_origin_labels_count(self, sort=False):
        unique, counts = np.unique(self.origin_y, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        if sort:
            sorted_counts = sorted(counts_dict.items(), key=operator.itemgetter(1))
            for k, v in sorted_counts:
                print('{} ({}) = {}'.format(k, self.decode(k), v))
        else:
            for k, v in counts_dict.items():
                print('{} ({}) = {}'.format(k, self.decode(k), v))
