import os

TRUE_CLUSTERS = "clustered2"
OUT_FILE = "clusters.csv"

data = []

for folder, _, images in os.walk(TRUE_CLUSTERS):
    cluster_id = os.path.basename(folder).split(" ")[-1]
    print("Found {} images in cluster #{}.".format(len(images), cluster_id))
    rows = [(image, cluster_id) for image in images]
    data += rows


print("Saving {} clusters".format(len(os.listdir(TRUE_CLUSTERS))))
with open(OUT_FILE, "w") as f:
    for row in data:
        csv_row = "{},{}\n".format(row[0], row[1])
        f.write(csv_row)
