import numpy as np

from darkocr.darkocr import DarkOCR


def predict(input_data):
    ocr = DarkOCR()
    return ocr.predict(input_data)


# testing
in_data = np.zeros((213, 56, 56, 1))
print(predict(in_data))


