import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

filename = "train-images.idx3-ubyte"
#filename = "t10k-images.idx3-ubyte"

with open(filename, "rb") as f:	#extracting image training data
	images = idx2numpy.convert_from_file(f)

print(images.shape)

plt.imshow(images[1], cmap="gray")
plt.title("First Image in MNIST")
plt.show()

filename = "train-labels.idx1-ubyte"
#filename = "t10k-labels.idx1-ubyte"

with open(filename, "rb") as f:	#extracting labels
	labels = idx2numpy.convert_from_file(f)	#convert from bin to numpy array

print(labels.shape)
print(f"label of first image: {labels[1]}",)

import cv2
import os


for i in range(9):
	os.makedirs(f"dataset/train/class_{i}", exist_ok=True)

for i in range(len(images)):
	cv2.imwrite(os.path.join(f"dataset/train/class_{labels[i]}/", f"img{i}.png"), images[i])


filename = "t10k-images.idx3-ubyte"

with open(filename, "rb") as f:	#extracting image training data
	images = idx2numpy.convert_from_file(f)

filename = "t10k-labels.idx1-ubyte"

with open(filename, "rb") as f:	#extracting labels
	labels = idx2numpy.convert_from_file(f)	#convert from bin to numpy array

for i in range(9):
	os.makedirs(f"dataset/test/class_{i}", exist_ok=True)

for i in range(len(images)):
	cv2.imwrite(os.path.join(f"dataset/test/class_{labels[i]}/", f"img{i}.png"), images[i])
