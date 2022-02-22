from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings, os
from glob import glob
from PIL import Image

# warnings.filterwarnings(action='ignore')

def load_img_to_numpy(path):
    
    path = path
    images = []
    labels = []
    
    for filename in glob(path +"*"):
        for img in glob(filename + "/*.jpg"):
            an_img = Image.open(img).convert('RGB').resize((100,100)) #read img
            img_array = np.array(an_img) #img to array
            images.append(img_array) #append array to training_images 
            label = filename.split('\\')[-1] #get label
            labels.append(label) #append label
            
    images = np.array(images)
    labels = np.array(labels)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels= le.fit_transform(labels)
    # labels = labels.reshape(-1,1)
    
    return images, labels

train_path = 'D:\_data\image_classification\cat_dog/training_set/'
test_path = 'D:\_data\image_classification\cat_dog/test_set/'

x_train,y_train = load_img_to_numpy(train_path)
x_test,y_test = load_img_to_numpy(test_path)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
