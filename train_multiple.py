import os
from keras.callbacks import ModelCheckpoint
from inception_v3_model import InceptionV3Model
from data_wrangler import DataWrangler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
CLUSTERED_PATH = "clustered"
LEARNING_RATE = 0.0001
EPOCHS = 25
BATCH_SIZE = 8
IMAGE_SIZE = (299, 299)

# autosave best Model
def gen_save_callback(boat_id):
    model_name = "weights." + str(boat_id) + ".h5"
    return ModelCheckpoint(model_name, monitor='val_acc', verbose = 1, save_best_only = True)

model = InceptionV3Model()
model.create_model(LEARNING_RATE, EPOCHS, BATCH_SIZE)

for boat_id in os.listdir(CLUSTERED_PATH):
    boat_path = os.path.join(CLUSTERED_PATH, boat_id)
    data_wrangler = DataWrangler(boat_path)
    data_wrangler.split()

    train_gen = data_wrangler.get_train_generator(IMAGE_SIZE, BATCH_SIZE)
    val_gen = data_wrangler.get_val_generator(IMAGE_SIZE, BATCH_SIZE)

    model.reset()
    model.fit(
        train_gen,
        val_gen,
        data_wrangler.num_train_samples,
        data_wrangler.num_val_samples,
        [gen_save_callback(boat_id)])

