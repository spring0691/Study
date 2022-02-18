from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import mnist, cifar10, cifar100

(x_train,y_train), (x_test,y_test) = cifar10.load_data()

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

model.summary()