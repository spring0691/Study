from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions     
# input에 전처리해준다.  incode 암호화, decode 복호화

model = ResNet50(weights='imagenet')