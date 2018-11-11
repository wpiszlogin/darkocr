from model import CNNModel
from data import *
from augment import *


class DarkOCR:
    def __init__(self):
        print('DarkOCR initialization...')
        self.data = ImageData()
        # reading data
        self.data.read_origin_data(pickle_path)
        self.model = CNNModel(image_dim, image_dim, 26)
        print('Complete')

    def show_origin_data(self, char='p'):
        # visualization
        self.data.show_origin_all_chars()
        self.data.show_origin_chars_data()
        self.data.show_origin_chars_data(char)

    def show_origin_data_statistics(self):
        self.data.show_origin_data_histogram()
        self.data.print_origin_labels_count()

    def save_data_set_to_png(self, path):
        self.data.save_data_set_to_png(path)

    def augment_folder(self, path, char_i=None, generated_count=50):
        pixels_mean = None
        if char_i is not None:
            path += '/' + str(char_i)
            pixels_mean_per_class = self.data.calc_pixels_mean()
            pixels_mean = pixels_mean_per_class[char_i]

        augment_folder(path, generated_count=generated_count, pixels_mean=pixels_mean)

    def fit_from_aug_folder(self, path=png_path):
        data_set = self.data.read_augmented_data_and_process(in_path=path, classes_count_int=4)
        self.fit(data_set)

    def fit_from_aug_pickle(self, aug_pickle_path=augmented_pickle_path):
        data_set = self.data.read_pickle(aug_pickle_path)
        self.fit(data_set)

    def fit(self, data_set):
        # single fold means standard cross-validation
        test_fold = 4

        (train_x, train_y), (test_x, test_y) = self.data.from_processed_data_to_training_set(
            data_set=data_set,
            test_fold=test_fold)

        self.model.fit(train_x, train_y, test_x, test_y)

    def predict(self, input_data):
        return self.model.predict(input_data)
