from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.models import load_model
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

class InceptionV3Model(object):

    def __init__(self):
        pass

    def reset(self):
        self.model.set_weights(self.init_weights)

    def create_model(self, learning_rate, epochs, batch_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        print('Loading InceptionV3 Weights ...')
        InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                                        input_tensor=None, input_shape=(299, 299, 3))

        print('Adding Average Pooling Layer and Softmax Output Layer ...')
        output = InceptionV3_notop.get_layer(index=-1).output  # Shape: (8, 8, 2048)
        output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(8, activation='softmax', name='predictions')(output)

        self.model = Model(InceptionV3_notop.input, output)

        optimizer = SGD(lr=self.learning_rate, momentum=0.9, decay=0.0, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.init_weights = self.model.get_weights()

    def load_model(self, weights_path):
        self.model = load_model(weights_path)

    def fit(self, train_data, val_data, nb_train_samples, nb_val_samples, callbacks):
        return self.model.fit_generator(
            train_data,
            samples_per_epoch=nb_train_samples,
            nb_epoch=self.epochs,
            validation_data=val_data,
            nb_val_samples=nb_val_samples,
            callbacks=callbacks)

    def predict(self, test_data, nb_test_samples):
        return self.model.predict_generator(test_data, nb_test_samples)

