import os
from scipy.misc import imread, imresize
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
import sys
TRAIN_PATH = "train"
IMAGE_SIZE = (50, 50)
OUT_PATH = "clustered"
N_CLUSTER = 25
BATCH_SIZE = 200

def create_boat_dirs(n_boats=N_CLUSTER, train_path=TRAIN_PATH, out_path=OUT_PATH):
    for boat_id in range(n_boats):
        for fish_id in os.listdir(train_path):
            clustered_dir = os.path.join(out_path, str(boat_id), fish_id)
            os.makedirs(clustered_dir)

def batch_generator(image_paths, batch_size=BATCH_SIZE):
    for i in range(len(image_paths) // batch_size):
        batch = image_paths[i*batch_size:(i+1)*batch_size]
        yield (i, load_to_array(batch))

def load_to_array(paths, new_size=IMAGE_SIZE):
    rgb_images = np.array([imresize(imread(path, flatten=True), new_size) for path in paths])
    images = rgb_images #np.sum(rgb_images, axis=3)
    flat_images = images.reshape(images.shape[0], -1)
    return flat_images


print("Creating new boat directories.")
try:
    create_boat_dirs()
except:
    pass

image_paths = [os.path.join(folder, file_name) for folder, _, file_names in os.walk(TRAIN_PATH) for file_name in file_names]

KM = KMeans(n_clusters=N_CLUSTER)
print("Loading images.")
images = load_to_array(image_paths)
print("Training k-means.")
clusters = KM.fit_predict(images)

# for i, batch in batch_generator(image_paths):
#     print("Clustering batch #{}.".format(i))
#     if i == 0:
#         clusters = KM.fit_predict(batch)
#     else:
#         clusters += KM.predict(batch)


# KM = MiniBatchKMeans(n_clusters=N_CLUSTER, batch_size=BATCH_SIZE)
# i = 0
# for i, batch in batch_generator(image_paths):
#     KM.partial_fit(batch)
#     print("Fit batch #{}.".format(i))
#     i += 1

# print("Clustering training set.")
# clusters = np.array([])
# for i, batch in batch_generator(image_paths):
#     clusters += KM.predict(batch)


print("Creating symlink from original train directory to new boat dirs.")
# Create symbolic link for each image to its boat directory
for i in range(len(image_paths)):
    src = image_paths[i]
    rootless = os.path.join(*src.split("\\")[1:]) # Gets rid of TRAIN_PATH root
    cluster = clusters[i]
    dst = os.path.join(OUT_PATH, str(cluster), rootless)
    os.link(src, dst)

print("Done.")
