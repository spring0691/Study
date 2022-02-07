from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np,time
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

