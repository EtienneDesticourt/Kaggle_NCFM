import os
import sys
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from data_wrangler import DataWrangler
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


IMG_SIZE = (299, 299)
BATCH_SIZE = 8
NB_CLASSES = 8
NB_TEST_SAMPLES = 1000
NB_AUGMENTATIONS = 5
TEST_DATA_DIR = 'test_stg1/'

if __name__ == "__main__":

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    data_wrangler = DataWrangler("")

    predictions = np.zeros((NB_TEST_SAMPLES, NB_CLASSES))
    model = load_model(model_path)

    for aug in range(NB_AUGMENTATIONS):
        test_generator = data_wrangler.get_test_generator(IMG_SIZE, BATCH_SIZE, TEST_DATA_DIR)
        predictions += model.predict_generator(test_generator, NB_TEST_SAMPLES)

    predictions /= NB_AUGMENTATIONS

    np.save(output_path, predictions)
