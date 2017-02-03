import os
import sys
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


IMG_WIDTH = 299
IMG_HEIGHT = 299
BATCH_SIZE = 8
NB_TEST_SAMPLES = 1000
NB_AUGMENTATION = 5
TEST_DATA_DIR = 'test_stg1/'

if __name__ == "__main__":

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    model = load_model(model_path)


    test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    for i in range(NB_AUGMENTATION):
        print('{}th augmentation for testing ...'.format(i))
        random_seed = np.random.random_integers(0, 100000)

        test_generator = test_datagen.flow_from_directory(
                TEST_DATA_DIR,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                shuffle = False, # Important !!!
                seed = random_seed,
                classes = None,
                class_mode = None)

        print('Begin to predict for testing data ...')
        current = model.predict_generator(test_generator, NB_TEST_SAMPLES)
        if i == 0:
            predictions = current
        else:
            predictions += current

    predictions /= NB_AUGMENTATION
    np.save(output_path, predictions)
