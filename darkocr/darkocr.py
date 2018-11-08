from model import CNNModel
from data import *
from augment import *


class DarkOCR:
    def __init__(self):
        self.data = ImageData()
        self.model = CNNModel(image_dim, image_dim, classes_count)

    def show_data(self, path=pickle_path, char='p'):
        # reading data
        self.data.read_reshaped_data(path)
        # visualization
        self.data.show_all_chars()
        self.data.show_chars_data()
        self.data.show_chars_data(char)

    def show_data_statistics(self, path=pickle_path):
        self.data.read_reshaped_data(path=path)
        self.data.show_data_histogram()
        self.data.print_labels_count()

    def augment_folder(self, generated_count=50, path=augment_path):
        pixels_mean_per_class = self.data.calc_pixels_mean()
        augment_folder(path, generated_count=generated_count, pixels_mean_per_class=pixels_mean_per_class)

    def predict(self, input_data):
        return self.model.predict(input_data)