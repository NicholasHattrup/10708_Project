import numpy as np

train_ids = []
valid_ids = []
test_ids = []
with open("train_ids.txt", "r") as f:
    for line in f:
        train_ids.append(line.replace("\n",""))
with open("valid_ids.txt", "r") as f:
    for line in f:
        train_ids.append(line.replace("\n",""))
with open("test_ids.txt", "r") as f:
    for line in f:
        train_ids.append(line.replace("\n",""))


