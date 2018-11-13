from darkocr.darkocr import DarkOCR


# modelling process
ocr = DarkOCR()
ocr.show_origin_data()
ocr.show_origin_data_statistics()
ocr.save_data_set_to_png('train_data/aug_test')
# it is better to do augment individually, to control it
ocr.augment_folder('train_data/aug_test/', 0, 50)
ocr.fit_from_aug_folder('train_data/aug_test/')

# process one char folder
from darkocr.darkocr import DarkOCR
ocr = DarkOCR()
ocr.augment_folder('D:/dev/projects/darkocr/train_data/png/', 19, 50)

# read augmented data, save to processed pickle file
from darkocr.data import ImageData
data = ImageData()
data_set = data.read_augmented_data_and_process(in_path='train_data/png/', out_path='train_data/aug_data36.pkl')
# test it
from PIL import Image
# [class][example][augmentation, 0 - origin]
Image.fromarray(data_set[0][0][0] * 255).show()

# generate training data
from darkocr.data import ImageData
data = ImageData()
train_set = data.from_processed_data_to_training_set(data_set=data.read_pickle('train_data/aug_data36.pkl'),
                                                     test_fold=4, ignore_class=30)
# show it
data.show_training_set()

# fit model
from darkocr.darkocr import DarkOCR
ocr = DarkOCR()
ocr.fit_from_aug_pickle(aug_pickle_path='train_data/aug_data36.pkl', test_fold=4)
ocr.model.plot_training()

# save model
ocr.model.save_model(fold=4)

# test model
from darkocr.darkocr import DarkOCR
from PIL import Image
from darkocr.data import ImageData
ocr = DarkOCR()
ocr.load_trained_models_group()
ocr.evaluate_by_image_folder('test_data/')

im = Image.open('test_data/1#3.png')
print(ImageData.decode(ocr.predict_image(im)[0]))