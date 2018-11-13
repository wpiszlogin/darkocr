from data import *
from darkocr.darkocr import DarkOCR


def predict(input_data):
    input_data = input_data.reshape(-1, image_dim, image_dim, 1)

    ocr = DarkOCR()
    ocr.load_trained_models_group()
    prediction = ocr.predict_from_group(input_data)

    # reshape to N * 1
    return prediction.reshape(-1, 1)


# testing
in_data = np.zeros((222, 3136))
print(predict(in_data))
