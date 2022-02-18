import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
_________________________________________________________________
'''
print("\n",model.weights,"\n\n ===========================================================\n")
'''
<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.11560118,  1.0606455 , -1.0812484 ]], dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=array([[ 0.03904426,  0.3300786 ],
[-0.97085786, -0.14864534],[ 0.6138109 , -0.98611933]], dtype=float32)>,
<tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>

<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=array([[ 0.9262482],[-0.4671206]], dtype=float32)>, 
<tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''
print(model.trainable_weights,"\n======================================================================")

print(len(model.weights))
print(len(model.trainable_weights))     # 각 레이어는 w+b 1세트 2개로 구성. len은 레이어 개수 * 2

model.trainable = False

print(len(model.weights))
print(len(model.trainable_weights))     # 각 레이어는 w+b 1세트 2개로 구성. len은 레이어 개수 * 2
