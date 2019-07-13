# script to generate text file for training, validation and testing test dataset
import random
import numpy as np
import os

dir = '/home/deepsight3/dev/deepsight/MultiView/data/XMLs'

# find the sequences
sequences = os.listdir(dir)
sequences.sort()

# read the frames
files = [[], [], [], []]
for idx, seq in enumerate(sequences):
    files[idx] = os.listdir(os.path.join(dir, seq))
    files[idx].sort()

# split data
train_percentage = 0.8
validation_percentage = 0.2

total_num = len(files[0])
train_num = np.round(total_num * train_percentage).astype(np.uint8)
validation_num = total_num - train_num

total_idx = np.arange(total_num)
train_idx = random.sample(range(total_num), train_num)
train_idx.sort()
validation_idx = np.delete(total_idx, train_idx)

# write to txt file
f = open("train_data.txt", "w+")
for index in train_idx:
    f.write(" %03d \n" % index)
f.close()
print("Train data successfully generated")

f = open("validation_data.txt", "w+")
for index in validation_idx:
    print(index)
    f.write(" %03d \n" % index)
f.close()
print("Validation data successfully generated")