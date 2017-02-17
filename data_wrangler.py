import os
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

TEMP_PATH = "temp"
TRAIN_DATA_DIR ="train"
VAL_DATA_DIR = "val"
TRAIN_SPLIT_PERCENT = 0.8

class DataWrangler(object):
    def __init__(self, data_path, temp_path=TEMP_PATH, train_split_percent=TRAIN_SPLIT_PERCENT):
        self.data_path = data_path
        self.temp_path = temp_path
        self.train_split_percent = TRAIN_SPLIT_PERCENT
        self.train_path = os.path.join(self.temp_path, TRAIN_DATA_DIR)
        self.val_path = os.path.join(self.temp_path, VAL_DATA_DIR)

    def build_split_directory_tree(self):
        try:
            shutil.rmtree(self.temp_path)
        except FileNotFoundError:
            pass
        os.mkdir(self.temp_path)
        os.mkdir(self.train_path)
        os.mkdir(self.val_path)
        for label in os.listdir(self.data_path):
            label_train_path = os.path.join(self.temp_path, TRAIN_DATA_DIR, label)
            label_val_path = os.path.join(self.temp_path, VAL_DATA_DIR, label)
            os.mkdir(label_train_path)
            os.mkdir(label_val_path)

    def create_symlinks(self, train_dir, temp_dir, files):
        for path in files:
            source = os.path.join(train_dir, path)
            dest = os.path.join(temp_dir, path)
            os.link(source, dest)

    def split(self):
        self.build_split_directory_tree()

        for label in os.listdir(self.data_path):
            full_label_path = os.path.join(self.data_path, label)
            data = os.listdir(full_label_path)
            if len(data) == 0: continue
            np.random.shuffle(data)

            self.num_train_samples = int(len(data) * self.train_split_percent)
            self.num_val_samples = len(data) - self.num_train_samples
            train_data = data[:self.num_train_samples]
            val_data = data[self.num_train_samples:]

            train_path = os.path.join(self.temp_path, TRAIN_DATA_DIR, label)
            self.create_symlinks(full_label_path, train_path, train_data)

            val_path = os.path.join(self.temp_path, VAL_DATA_DIR, label)
            self.create_symlinks(full_label_path, val_path, val_data)

    def split_with_file(self, file_path):
        self.build_split_directory_tree()

        train_files = []
        val_files = []
        with open(file_path, "r") as f:
            for line in f:
                file_name, train_or_val = line.split(",")
                if train_or_val == "train\n":
                    train_files.append(file_name)
                else:
                    val_files.append(file_name)

        for label in os.listdir(self.data_path):
            full_label_path = os.path.join(self.data_path, label)
            data = os.listdir(full_label_path)
            if len(data) == 0: continue

            train_data = [row for row in data if row in train_files]
            val_data = [row for row in data if row in val_files]

            self.num_train_samples = len(train_data)
            self.num_val_samples = len(val_data)

            train_path = os.path.join(self.temp_path, TRAIN_DATA_DIR, label)
            self.create_symlinks(full_label_path, train_path, train_data)

            val_path = os.path.join(self.temp_path, VAL_DATA_DIR, label)
            self.create_symlinks(full_label_path, val_path, val_data)


    def get_train_generator(self, image_size, batch_size):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            rotation_range=10.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        train_generator = train_datagen.flow_from_directory(self.train_path,
            target_size = image_size,
            batch_size = batch_size,
            classes = os.listdir(self.train_path))
        return train_generator

    def get_val_generator(self, image_size, batch_size, augment=False):
        val_datagen = ImageDataGenerator(rescale=1./255)
        validation_generator = val_datagen.flow_from_directory(self.val_path,
            target_size=image_size,
            batch_size=batch_size,
            classes = os.listdir(self.train_path))
        return validation_generator

    def get_test_generator(self, image_size, batch_size, test_path):
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        random_seed = np.random.random_integers(0, 100000)

        test_generator = test_datagen.flow_from_directory(
                test_path,
                target_size=image_size,
                batch_size=batch_size,
                shuffle = False, # Important !!!
                seed = random_seed,
                classes = None,
                class_mode = None)
        return test_generator


