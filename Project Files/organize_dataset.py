import os
import shutil
import pandas as pd


print("Reading CSV...")

labels = pd.read_csv("labels.csv")



print("Total rows in CSV:", len(labels))

os.makedirs("dataset/train", exist_ok=True)

count = 0

for index, row in labels.iterrows():
    breed = row["breed"]
    image_id = row["id"] + ".jpg"

    source = os.path.join("train", image_id)
    destination_folder = os.path.join("dataset/train", breed)

    os.makedirs(destination_folder, exist_ok=True)

    destination = os.path.join(destination_folder, image_id)

    if os.path.exists(source):
        shutil.copy(source, destination)
        count += 1

print("Total images copied:", count)
print("Done.")
print(labels.isnull().sum())
