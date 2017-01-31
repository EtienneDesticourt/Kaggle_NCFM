import os
import shutil
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from inception_v3_model import InceptionV3Model
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time

# Fitting parameters
LEARNING_RATE = 0.0001
EPOCHS = 25
BATCH_SIZE = 8
NB_FOLDS = 5

# Data parameters
IMG_WIDTH = 299
IMG_HEIGHT = 299
NB_TRAIN_SAMPLES = 294
NB_VAL_SAMPLES = 79
ROOT_DATA_DIR = 'train_sample'
TEMP_SPLIT_DIR = 'temp'
TRAIN_DATA_DIR = os.path.join(TEMP_SPLIT_DIR, 'train_split')
VAL_DATA_DIR = os.path.join(TEMP_SPLIT_DIR, 'val_split')
FISH_NAMES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

def load_path_data(train_dir):
    paths = []
    labels = []
    for fish_path, _, images in os.walk(train_dir):
        fish_name = os.path.basename(fish_path)
        paths += [os.path.join(fish_name, img) for img in images]
        labels += [fish_name] * len(images)
    return np.array(paths), np.array(labels)

def reset_temp_dirs():
    try:
        shutil.rmtree(TEMP_SPLIT_DIR)
    except FileNotFoundError:
        pass
    os.mkdir(TEMP_SPLIT_DIR)
    os.mkdir(TRAIN_DATA_DIR)
    os.mkdir(VAL_DATA_DIR)
    for fish in FISH_NAMES:
        train_path = os.path.join(TRAIN_DATA_DIR, fish)
        val_path = os.path.join(VAL_DATA_DIR, fish)
        os.mkdir(train_path)
        os.mkdir(val_path)

def link_to_temp_dir(train_dir, temp_dir, files):
    for path in files:
        source = os.path.join(train_dir, path)
        dest = os.path.join(temp_dir, path)
        os.link(source, dest)


# autosave best Model
def gen_save_callback(index):
    model_name = "weights" + str(index) + ".h5"
    return ModelCheckpoint(model_name, monitor='val_acc', verbose = 1, save_best_only = True)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)




paths, labels = load_path_data(ROOT_DATA_DIR)
skf = StratifiedKFold(n_splits=NB_FOLDS, shuffle=True)

fold = 1
for train_index, val_index in skf.split(paths, labels):

    # Copy train data to split folders for generators
    reset_temp_dirs()
    train_paths = paths[train_index]
    link_to_temp_dir(ROOT_DATA_DIR, TRAIN_DATA_DIR, train_paths)
    val_paths = paths[val_index]
    link_to_temp_dir(ROOT_DATA_DIR, VAL_DATA_DIR, val_paths)

    # Create generators with new split
    train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR,
        target_size = (IMG_WIDTH, IMG_HEIGHT),
        batch_size = BATCH_SIZE,
        classes = FISH_NAMES)

    validation_generator = val_datagen.flow_from_directory(VAL_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        classes = FISH_NAMES)

    # Train model with new split
    save_best_model = gen_save_callback(fold)
    inception_v3_model = InceptionV3Model()
    inception_v3_model.create_model(LEARNING_RATE, EPOCHS, BATCH_SIZE)
    inception_v3_model.fit(train_generator, validation_generator, NB_TRAIN_SAMPLES, NB_VAL_SAMPLES, [save_best_model])

    fold += 1

