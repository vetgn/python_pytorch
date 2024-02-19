import numpy as np
import matplotlib.pyplot as plt

data_file = open("dataset1/mnist_test.csv", "r")
data_list = data_file.readlines()
data_file.close()
all_values = data_list[0].split(',')
image_array = np.
plt.im