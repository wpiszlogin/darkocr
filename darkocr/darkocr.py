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

        print('before {}'.format(pixels_mean))
        augment_folder(path, generated_count=generated_count, pixels_mean=pixels_mean)

    def predict(self, input_data):
        return self.model.predict(input_data)
