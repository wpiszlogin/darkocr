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
