
'''
install following package to do machine learning
pip install tensorflow
pip install keras
pip install numpy
pip install pillow
pip install scipy
'''

'''
This section is to make the model, it only run one time
'''

# Importing all necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


'''
Here, the train is the train dataset directory.
validation is the directory for validation data.
nb_train_samples is the total number of train samples.
nb_validation_samples is the total number of validation samples.
'''

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples =288
nb_validation_samples = 288
epochs = 5
batch_size = 10
img_width, img_height = 640, 480
'''
This part is to check the data format i.e the RGB channel is coming first or last so,
whatever it may be, the model will check first and then input shape will be fed accordingly.
'''
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

'''
Conv2D is the layer to convolve the image into multiple images 
Activation is the activation function. 
MaxPooling2D is used to max pool the value from the given size matrix and same is used for the next 2 layers. then, Flatten is used to flatten the dimensions of the image obtained after convolving it. 
Dense is used to make this a fully connected model and is the hidden layer. 
Dropout is used to avoid overfitting on the dataset. 
Dense is the output layer contains only one neuron which decide to which category image belongs.
'''

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


'''
Compile function is used here that involve the use of loss, optimizers and metrics.
Here loss function used is binary_crossentropy, optimizer used is rmsprop.
'''

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

'''
Now, the part of dataGenerator comes into the figure. In which we have used: 

ImageDataGenerator that rescales the image, applies shear in some range,
zooms the image and does horizontal flipping with the image.
This ImageDataGenerator includes all possible orientation of the image.

train_datagen.flow_from_directory is the function that is used
to prepare data from the train_dataset directory,
Target_size specifies the target size of the image.

test_datagen.flow_from_directory is used to prepare test data for the model and
all is similar as above.

fit_generator is used to fit the data into the model made above,
other factors used are steps_per_epochs tells us about the number of times
the model will execute for the training data. 
epochs tells us the number of times model will be trained in forward and backward pass. 
validation_data is used to feed the validation/test data into the model. 
validation_steps denotes the number of validation/test samples.
'''
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

classes = train_generator.class_indices
print('classes are ', classes)

'''
At last, we can also save the model.
'''

model.save('model_saved.h5')



'''
This secton is Loading Model and Prediction

Load Model with “load_model”
Convert Images to Numpy Arrays for passing into ML Model
Print the predicted output from the model.
'''
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np

from keras.models import load_model

import os
#it is the directory of images that were taken in ISS
dir= r'D:\F\Home\CodingTutorial\astropi\2022\jwl\jwa\jwa\jwa'

#the image file name list are saved in picture_name.txt
img_files = 'picture_name.txt'

#initial Sea and Land, they are used to count the the amount of of Sea and Land
Sea = 0
Land = 0

# read the file name and recorgnize them one by one 
with open(img_files) as f:
    for line in f:
        img_name = f.readline().strip()
        print(img_name)
        img_name = os.path.join(dir,img_name)
        image = load_img(img_name, target_size=(640, 480))
        model = load_model('model_saved.h5')
        img = np.array(image)
        img = img / 255.0
        img = img.reshape(1,640,480,3)

        predict = (model.predict(img) > 0.5).astype("int32")
        print(predict)

#  If the image is recorgnized as land,  Land increase 1
        if predict[0][0] == 0:
            print('it is land')
            Land = Land + 1
# If the image is recorgnized as sea,  Sea increase 1            
        if predict[0][0] == 1:
            print('it is sea')
            Sea = Sea + 1

print('Land is ', Land)
print('Sea is ', Sea)

#under this line, Append Sea and Land to file 'LandAndSea.txt'
f = open('LandAndSea.txt','a')
f.write('Land is ' + str(Land) + '\n')
f.write('Sea is ' + str(Sea) + '\n')
f.close()
