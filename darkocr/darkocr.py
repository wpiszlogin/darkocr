import glob
from PIL import Image

from model import CNNModel
from data import *
from augment import *


class DarkOCR:
    def __init__(self):
        print('DarkOCR initialization...')
        self.data = ImageData()
        # reading data
        self.data.read_origin_data(pickle_path)
        self.model = CNNModel(image_dim, image_dim, classes_count)
        self.models_fold = [CNNModel(image_dim, image_dim, classes_count) for i in range(fold_count)]
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

    def fit_from_aug_pickle(self, aug_pickle_path=augmented_pickle_path, test_fold=4):
        data_set = self.data.read_pickle(aug_pickle_path)
        self.fit(data_set, test_fold=test_fold)

    def fit(self, data_set, test_fold=4):
        (train_x, train_y), (test_x, test_y) = self.data.from_processed_data_to_training_set(
            data_set=data_set,
            test_fold=test_fold,
            ignore_class=30)

        self.model.fit(train_x, train_y, test_x, test_y)

    def load_trained_models_group(self):
        print('Loading models group...')
        for fold in range(fold_count):
            self.models_fold[fold].load_model(fold=fold)
        print('Done')

    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        return np.argmax(prediction, axis=1)

    def predict_from_fold(self, input_data, fold=4):
        prediction = self.models_fold[fold].predict(input_data)
        return np.argmax(prediction, axis=1)

    def predict_from_group(self, input_data):
        print('Making prediction...')
        prediction_votes = np.zeros((len(input_data), classes_count))
        for fold in range(fold_count):
            prediction_votes += self.models_fold[fold].predict(input_data)

        return np.argmax(prediction_votes, axis=1)

    def predict_image(self, im, decode=False):
        im = im.convert("L")
        ia = np.array(im, dtype='d')

        ia = ia / 255
        ia = ia.reshape(-1, image_dim, image_dim, 1)
        prediction = self.predict_from_group(ia)
        if decode:
            prediction = ImageData.decode(prediction[0])

        return prediction

    def evaluate_by_image_folder(self, path):
        correct_count = 0
        examples_count = 0

        answers_counter = [0] * classes_count
        correct_counter = [0] * classes_count

        for im_path in glob.glob(path + '/*.png'):
            hash_i = im_path.rfind("#")
            label = im_path[hash_i + 1]

            im = Image.open(im_path)
            prediction = self.predict_image(im, decode=True)

            if label == prediction:
                correct_count += 1
                correct_counter[ImageData.encode(label)] += 1

            answers_counter[ImageData.encode(prediction)] += 1
            examples_count += 1

        accuracy = 100 * correct_count / examples_count
        print('Results: {:.2f}%'.format(accuracy))
        for i in range(len(answers_counter)):
            print('{} ({}). correct: {}, answers count: {}'.format(
                i, ImageData.decode(i), correct_counter[i], answers_counter[i]))
        return accuracy