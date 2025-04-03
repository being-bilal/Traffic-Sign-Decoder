import os 
import json
import matplotlib.pyplot as plt
import seaborn as sns

path = "dataset/Train"
classes = os.listdir(path=path)
data = dict()

mapping = open("assets/mapping.json", "r")
mapping = json.load(mapping)


for cls in classes:
    cls_path = os.path.join(path, cls)
    class_name = mapping[str(cls)]
    data[class_name] = len(os.listdir(cls_path))

plt.figure(figsize=(18,8))
sns.barplot(x=list(data.keys()), y=list(data.values()), palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("Traffic Sign Type")
plt.ylabel("Number of Images")
plt.title("Traffic Sign Class Distribution")
plt.show()