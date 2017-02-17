import os
from boat_clusterer import create_boat_dirs, sort_by_cluster

BOAT_CLASSIFIER_MODEL_FILE = "boat_classifier_22.h5"
RAW_PATH = "train"
CLUSTERED_PATH = "clustered2"
OUT_PATH = "clustered3"

if __name__ == "__main__":

    main_clusters = [int(i) for i in os.listdir(CLUSTERED_PATH)]
    unclustered_cluster_id = max(main_clusters) + 1
    clusters = {}
    with open("clusters.csv") as f:
        for line in f:
            image, cluster = line[:-1].split(",")
            cluster = int(cluster)
            if cluster not in main_clusters:
                cluster = unclustered_cluster_id
            clusters[image] = cluster

    cluster_list = []
    image_paths = [os.path.join(folder, image) for folder, _, images in os.walk(RAW_PATH) for image in images]

    def get_cluster(image):
        try:
            return clusters[image]
        except KeyError:
            return unclustered_cluster_id

    clusters = [get_cluster(image) for folder, _, images in os.walk(RAW_PATH) for image in images]


    create_boat_dirs(boat_classes=main_clusters + [unclustered_cluster_id], out_path=OUT_PATH)
    sort_by_cluster(image_paths, clusters, out_path=OUT_PATH)

