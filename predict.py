from darkocr.darkocr import DarkOCR
import numpy as np


def predict(input_data):
    ocr = DarkOCR()
    return ocr.predict(input_data)


# testing
in_data = np.zeros((213, 56, 56, 1))
print(predict(in_data))

ocr = DarkOCR()
ocr.show_data()
ocr.show_data_statistics()
ocr.augment_folder()

# from pickle data set to png
from darkocr.data import ImageData
d = ImageData()
d.read_reshaped_data('train_data/train.pkl')
d.save_data_set_to_png('train_data/png/')

# process one char folder
from darkocr.darkocr import DarkOCR
ocr = DarkOCR()
ocr.augment_folder(50, 'D:/dev/projects/darkocr/train_data/png/0', 0)

png_path = 'train_data/png/'
augmented_pickle_path = 'train_data/aug_data.pkl'

# read augmented data
from darkocr.data import ImageData
d = ImageData()
dataset = d.read_augmented_data_set(in_path='train_data/png/', out_path='train_data/aug_data.pkl', classes_count_int=4)

from PIL import Image, ImageFilter
Image.fromarray(dataset[0][0][0] * 255).show()

# generate training data
from darkocr.data import ImageData
d = ImageData()
trainset = d.from_aug_pickle_to_training_set(aug_pickle_path='train_data/aug_data.pkl', test_fold=4)
d.show_training_set()

# fit model
from darkocr.darkocr import DarkOCR
ocr = DarkOCR()
ocr.fit_from_aug_pickle()
