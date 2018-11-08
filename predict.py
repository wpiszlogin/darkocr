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
