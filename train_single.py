import os
import sys
from keras.callbacks import ModelCheckpoint
from inception_v3_model import InceptionV3Model
from data_wrangler import DataWrangler


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
CLUSTERED_PATH = "clustered"
LEARNING_RATE = 0.0001
EPOCHS = 25
BATCH_SIZE = 8
IMAGE_SIZE = (299, 299)
RAW_PATH = "train"
NUM_AUGMENTATIONS = 5

if __name__ == "__main__":


    raw_data_path = sys.argv[1]

    data_wrangler = DataWrangler(raw_data_path)
    #data_wrangler.split()
    data_wrangler.split_with_file("original_split.split")

    # autosave best Model
    def get_save_callback(model_name):
        save_callback = ModelCheckpoint(model_name, monitor='val_acc', verbose = 1, save_best_only = True)
        return save_callback


    model = InceptionV3Model()
    model.create_model(LEARNING_RATE, EPOCHS, BATCH_SIZE)

    for aug in range(NUM_AUGMENTATIONS):

        train_gen = data_wrangler.get_train_generator(IMAGE_SIZE, BATCH_SIZE)
        val_gen = data_wrangler.get_val_generator(IMAGE_SIZE, BATCH_SIZE)

        save_callback = get_save_callback("InceptionV3_" + str(aug) + ".h5")

        model.reset()
        model.fit(
            train_gen,
            val_gen,
            data_wrangler.num_train_samples,
            data_wrangler.num_val_samples,
            [save_callback])
