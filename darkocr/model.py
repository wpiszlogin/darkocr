import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense


class CNNModel:
    def __init__(self, im_width, im_height, classes_count):
        self.in_width = im_width
        self.in_height = im_height
        self.classes_count = classes_count

        self.history = None

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
        self.history = self.model.fit(train_x, train_y, batch_size=512, validation_data=(test_x, test_y),
                                      epochs=30, verbose=1)

    def predict(self, data_x):
        return self.model.predict(data_x)

    def plot_training(self, key='loss'):
        plt.figure(figsize=(16, 10))
        name = str(self.classes_count) + " classes"

        val = plt.plot(self.history.epoch, self.history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(self.history.epoch, self.history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()

        plt.xlim([0, max(self.history.epoch)])

        fig = plt.gcf()
        fig.canvas.set_window_title('Training plot')
        plt.show()
