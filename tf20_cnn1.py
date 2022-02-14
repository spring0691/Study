import tensorflow as tf, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
tf.compat.v1.set_random_seed(66)

#1. 데이터 
from keras.datasets import mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()

