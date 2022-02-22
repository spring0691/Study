from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

print(model.layers)
# [<keras.layers.core.dense.Dense object at 0x000001C9E8936A90>, <keras.layers.core.dense.Dense object at 0x000001C98C865340>,
# <keras.layers.core.dense.Dense object at 0x000001C992D9B820>]