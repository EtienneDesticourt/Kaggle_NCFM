import os
from PIL import Image

print("Loading masks.")
masks = {}
for mask in os.listdir("masks"):
    full_path = os.path.join("masks", mask)
    cluster = mask.split("mask")[1].split(".")[0]
    masks[cluster] = Image.open(full_path)

print("Loading clusters.")
clusters = {}
with open("clusters.csv", "r") as f:
    for line in f:
        image, cluster = line[:-1].split(",")
        clusters[image] = cluster

print("Overlaying existing masks over clustered images.")
for folder, _, images in os.walk("train"):
    for image in images:
        try:
            cluster = clusters[image]
            mask = masks[cluster]
        except KeyError:
            continue
        full_path = os.path.join(folder, image)
        boat = Image.open(full_path)
        boat.paste(mask, (0, 0), mask)
        boat.save(full_path)

print("Done.")


