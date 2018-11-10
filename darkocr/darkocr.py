from model import CNNModel
from data import *
from augment import *


class DarkOCR:
    def __init__(self):
        print('DarkOCR initialization...')
        self.data = ImageData()
        # reading data
        self.data.read_reshaped_data(pickle_path)
        self.model = CNNModel(image_dim, image_dim, classes_count)

    def show_data(self, path=pickle_path, char='p'):
        # visualization
        self.data.show_all_chars()
        self.data.show_chars_data()
        self.data.show_chars_data(char)

    def show_data_statistics(self, path=pickle_path):
        self.data.show_data_histogram()
        self.data.print_labels_count()

    def augment_folder(self, generated_count=50, path=augment_path, char_i=None):
        pixels_mean = None
        if char_i is not None:
            pixels_mean_per_class = self.data.calc_pixels_mean()
            pixels_mean = pixels_mean_per_class[char_i]

        augment_folder(path, generated_count=generated_count, pixels_mean=pixels_mean)

    def fit_from_aug_pickle(self, aug_pickle_path=augmented_pickle_path):
        # single fold means standard cross-validation
        test_fold = 4

        (train_x, train_y), (test_x, test_y) = self.data.from_aug_pickle_to_training_set(
            aug_pickle_path=aug_pickle_path,
            test_fold=test_fold)

        self.model.fit(train_x, train_y, test_x, test_y)

    def predict(self, input_data):
        return self.model.predict(input_data)
