from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense


class CNNModel:
    def __init__(self, im_width, im_height, classes_count):
        self.in_width = im_width
        self.in_height = im_height
        self.classes_count = classes_count

        # build convolution network model
        self.model = keras.Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                              input_shape=(im_width, im_height, 1)))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(classes_count, activation='softmax'))

        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self, train_x, train_y, test_x, test_y):
        self.model.fit(train_x, train_y, batch_size=512, validation_data=(test_x, test_y), epochs=5, verbose=1)

    def predict(self, data_x):
        return self.model.predict(data_x)
