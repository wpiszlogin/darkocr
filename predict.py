import pickle
import numpy as np
import matplotlib.pyplot as plt

labels_count = 36
image_dim = 56


def predict(input_data):
    output_data = 'done'
    return output_data


def read_data(path):
    with open(path, "rb") as pickle_data:
        train_data = pickle.load(pickle_data)
        pickle_data
        return train_data


def decode(char_i, do_decode=True):
    decoding_list = [
        '6', 'P', '0', 'V', 'W', '3', 'A', '8', 'T', '1',
        '0', '9', 'H', 'R', 'N', '7', 'K', 'L', 'G', '4',
        'Y', 'C', 'E', 'J', '5', 'I', 'S', '2', 'F', 'Z',
        'N', 'Q', 'M', 'B', 'D', 'U']
    if do_decode:
        return decoding_list[char_i]
    else:
        return char_i


def find_indexes(data_y, char, count):
    # find indexes where label == char
    char_args = np.argwhere(data_y == char)
    char_args = [i for i, j in char_args]
    # select random elements
    return np.random.choice(char_args, count)


def show_all_chars(data_x, data_y):
    examples_count = 10
    do_decode = True

    col = 0
    plt.figure(figsize=(20, 9))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for char in range(labels_count):
        char_indexes = find_indexes(data_y, char, examples_count)
        # show them
        row = 0
        for label_index in char_indexes:
            # goes through columns
            plt.subplot(examples_count, labels_count, row * labels_count + col + 1)
            # we need to reshape to image_dim x image_dim
            image = data_x[label_index].reshape((image_dim, image_dim))
            plt.imshow(image, cmap=plt.cm.binary)
            plt.xlabel(decode(data_y[label_index][0], do_decode))
            plt.xticks([])
            plt.yticks([])
            row += 1
        col += 1
    fig = plt.gcf()
    fig.canvas.set_window_title('All characters')


def show_chars_data(data_x, data_y, char=None):
    cols_count = 27
    rows_count = 10
    do_decode = True

    char_indexes = []
    if char is not None:
        char_indexes = find_indexes(data_y, char, cols_count * rows_count)

    plt.figure(figsize=(20, 11))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for i in range(cols_count * rows_count):
        index = i
        if i < len(char_indexes):
            index = char_indexes[i]

        plt.subplot(rows_count, cols_count, i + 1)
        image = data_x[index].reshape((image_dim, image_dim))
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(decode(data_y[index][0], do_decode))
        plt.xticks([])
        plt.yticks([])
    fig = plt.gcf()
    fig.canvas.set_window_title('Characters data')


# reading data
train_x, train_y = read_data("train_data/train.pkl")
show_all_chars(train_x, train_y)
show_chars_data(train_x, train_y, char=0)







