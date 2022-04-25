import os
import numpy as np

files = os.listdir("xyz")


idx = np.random.permutation(len(files))
idx = idx.tolist()
valid_ids = [files[i] for i in idx[0:10000]]
test_ids = [files[i] for i in idx[10000:20000]]
train_ids = [files[i] for i in idx[20000:]]

with open("valid_ids.txt", "w") as f:
    for i in valid_ids:
        f.write(i+"\n")

with open("test_ids.txt", "w")	as f:
    for	i in test_ids:
        f.write(i+"\n")

with open("train_ids.txt", "w")	as f:
    for	i in train_ids:
        f.write(i+"\n")
