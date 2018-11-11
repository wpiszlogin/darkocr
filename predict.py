from darkocr.darkocr import DarkOCR
import numpy as np


def predict(input_data):
    ocr = DarkOCR()
    return ocr.predict(input_data)


# testing
in_data = np.zeros((213, 56, 56, 1))
print(predict(in_data))

# modelling process
ocr = DarkOCR()
ocr.show_origin_data()
ocr.show_origin_data_statistics()
ocr.save_data_set_to_png('train_data/aug_test')
# it is better to do augmentation individually, to control it
ocr.augment_folder('train_data/aug_test/', 0, 50)
ocr.fit_from_aug_folder('train_data/aug_test/')

# process one char folder
from darkocr.darkocr import DarkOCR
ocr = DarkOCR()
ocr.augment_folder('D:/dev/projects/darkocr/train_data/png/', 19, 50)

# read augmented data
from darkocr.data import ImageData
data = ImageData()
data_set = data.read_augmented_data_and_process(in_path='train_data/png/', out_path='train_data/aug_data.pkl',
                                                classes_count_int=4)
# test it
from PIL import Image
Image.fromarray(data_set[0][0][0] * 255).show()

# generate training data
from darkocr.data import ImageData
data = ImageData()
train_set = data.from_processed_data_to_training_set(data_set=data.read_pickle('train_data/aug_data.pkl'), test_fold=4)
data.show_training_set()

# fit model
from darkocr.darkocr import DarkOCR
ocr = DarkOCR()
ocr.fit_from_aug_pickle()
