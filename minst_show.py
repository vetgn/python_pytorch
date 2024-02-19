from dataset.mnist import load_mnist
import numpy as np
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(img)
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# print(x_train, t_train, x_test, t_test)
img = x_train[10]
label = t_train[10]
# print(label)
# print(img.shape)
img = img.reshape(28, 28)
img_show(img)
