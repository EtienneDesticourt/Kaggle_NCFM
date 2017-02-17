import os
from keras.models import load_model
from data_wrangler import DataWrangler
import numpy as np

def save_split_to_file():
    files = []
    for folder, _, images in os.walk("original_split\\train_split"):
        files += ["{},train".format(f) for f in images]

    for folder, _, images in os.walk("original_split\\val_split"):
        files += ["{},val".format(f) for f in images]

    with open("original_split.split", "w") as f:
        f.write("\n".join(files))


def cluster_from_file():
    clusters = {}
    with open("clusters.csv", "r") as f:
        added_clusters = []
        for line in f:
            name, cluster = line.split(",")
            cluster = cluster[:-1]
            clusters[name] = cluster
            if cluster not in added_clusters:
                added_clusters.append(cluster)
                path = os.path.join("clustered2", str(cluster))
                os.mkdir(path)

    for folder, _, images in os.walk("train"):
        for image in images:
            source = os.path.join(folder, image)
            dest = os.path.join("clustered2", str(clusters[image]), image)
            os.link(source, dest)


def uncluster(in_path, out_path):
    for folder, _, images in os.walk(in_path):
        for image in images:
            src = os.path.join(folder, image)
            dst = os.path.join(out_path, image)
            os.link(src, dst)

def simple_predict(model_path, data_path, nb_samples):
    model = load_model(model_path)
    dw = DataWrangler("")
    test_generator = dw.get_test_generator((299, 299), 8, data_path)
    prediction = model.predict_generator(test_generator, nb_samples)
    clusters = np.argmax(prediction, axis=1)
    cluster_certainty = np.amax(prediction, axis=1)
    clusters[cluster_certainty < 0.5] = 45
    images = np.array(os.listdir("unclustered\\New folder"))
    print(images[cluster_certainty < 0.5].shape)
    print(images[cluster_certainty < 0.4].shape)
    print(images[cluster_certainty < 0.3].shape)
    print(images[cluster_certainty < 0.2].shape) #This un
    print(images[cluster_certainty < 0.1].shape)
    for i in images[cluster_certainty < 0.2]:
        print(i)

if __name__ == "__main__":
    #uncluster("train", "unclustered")
    simple_predict("boat_classifier_22.h5", "unclustered", 3759)
