# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:50:07 2021

@author: User
"""

from datetime import datetime
import tensorflow as tf
from keras import backend as K
import os 
import numpy as np 
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
import random
import larq as lq

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
       # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

#set image height, width and channel
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
start_neuron = 16

seed = 42
np.random.seed = seed


os.chdir('E:/unet/ictpet dataset/images')
list = os.listdir('E:/unet/ictpet dataset/images')
print(list)

mask = []
image =[]

for filename in list:
    if filename.endswith('.jpg'):
        image.append(filename)
    if filename.endswith('.png'):
        mask.append(filename)

image.sort()
mask.sort()

#only take up to 1000 data
image = image[:1000]
mask = mask[:1000]

XTrain = np.zeros((1000,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), dtype = np.uint8)
YTrain = np.zeros((1000,IMG_HEIGHT,IMG_WIDTH,1), dtype = np.bool)

# for file in image:
#     index = image.index(file)
#     dir_image = os.path.join('E:/unet/ictpet dataset/images/', file)
#     img = imread(dir_image)[:,:,:IMG_CHANNELS]
#     img = resize(img, (IMG_HEIGHT,IMG_WIDTH), mode= 'constant',preserve_range=True)
#     XTrain[n] = img

for n, id_ in tqdm(enumerate(image), total = len(image)):
    #index = image.index(file)
    dir_image = os.path.join('E:/unet/ictpet dataset/images/', id_)
    img = imread(dir_image)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT,IMG_WIDTH), mode= 'constant',preserve_range=True)
    XTrain[n] = img

#maskarray = np.zeros((IMG_HEIGHT,IMG_WIDTH,1), dtype = np.bool)
#maskarray = np.zeros((IMG_HEIGHT,IMG_WIDTH,1), dtype = np.bool)
for n, id_ in tqdm(enumerate(mask), total =len(mask)):
    dir_mask = os.path.join('E:/unet/ictpet dataset/images/', id_)
    mask = imread(dir_mask)
    maskextracted = np.expand_dims(resize(mask,(IMG_HEIGHT,IMG_WIDTH),mode ='constant'
                                               ,preserve_range = True), axis = -1)
   # maskmax = np.maximum(mask, maskextracted)
    YTrain[n] = maskextracted

image_x = random.randint(0, len(image))
imshow(XTrain[image_x])
plt.show()
imshow(np.squeeze(YTrain[image_x]))
plt.show()

#Build model

#input
inputs = tf.keras.layers.Input((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))

#divide input rgb values by 255
newinput = tf.keras.layers.Lambda(lambda x: x / 127.5 - 1)(inputs)

#all quantized layers except the first will use the same settings, in the first layer 
#will be only quantize the weights
kwargs = dict(input_quantizer ="ste_sign",
              kernel_quantizer ="ste_sign",
              kernel_constraint = "weight_clip",
              use_bias = False)

#first layer will we only quantized the weights and not the input
conv1 = lq.layers.QuantConv2D(start_neuron *1,(3,3),kernel_quantizer ="ste_sign",kernel_constraint ="weight_clip",use_bias = False, activation='relu', padding ='same',pad_values = 1.0)(newinput)
#contracting path  start 2 convolutional layers with feature space of 16 kernel size of 3x3
#batch normalization after each convolutional layer 
conv1 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(conv1)
conv1 = tf.keras.layers.Dropout(0.1)(conv1)
#convolutional layers quantized the weights and the input  
conv1 = lq.layers.QuantConv2D(start_neuron *1,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(conv1)
conv1 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(conv1)
pool1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)
pool1 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(pool1)

#conv2 = tf.keras.layers.Conv2D(start_neuron * 2, (3,3),activation ='relu',kernel_initializer = 'he_normal', padding = 'same')(pool1)
#conv2 = tf.keras.layers.Dropout(0.1)(conv2)
#conv2 = tf.keras.layers.Conv2D(start_neuron * 2,(3,3), activation ='relu', kernel_initializer = 'he_normal', padding ='same')(conv2)
#pool2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)

conv2 = lq.layers.QuantConv2D(start_neuron *2,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(pool1)
#contracting path  start 2 convolutional layers with feature space of 16 kernel size of 3x3
#batch normalization after each convolutional layer 
conv2 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(conv2)
conv2 = tf.keras.layers.Dropout(0.1)(conv2)
#convolutional layers quantized the weights and the input  
conv2 = lq.layers.QuantConv2D(start_neuron *2,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(conv2)
conv2 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(conv2)
pool2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)
pool2 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(pool2)

conv3 = lq.layers.QuantConv2D(start_neuron *4,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(pool2)
#contracting path  start 2 convolutional layers with feature space of 16 kernel size of 3x3
#batch normalization after each convolutional layer 
conv3 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(conv3)
conv3 = tf.keras.layers.Dropout(0.1)(conv3)
#convolutional layers quantized the weights and the input  
conv3 = lq.layers.QuantConv2D(start_neuron *4,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(conv3)
conv3 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(conv3)
pool3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)
pool3 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(pool3)

#conv3 = tf.keras.layers.Conv2D(start_neuron * 4, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(pool2)
#conv3 = tf.keras.layers.Dropout(0.2)(conv3)
#conv3 = tf.keras.layers.Conv2D(start_neuron * 4, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(conv3)
#pool3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)  

conv4 = lq.layers.QuantConv2D(start_neuron *8,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(pool3)
#contracting path  start 2 convolutional layers with feature space of 16 kernel size of 3x3
#batch normalization after each convolutional layer 
conv4 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(conv4)
conv4 = tf.keras.layers.Dropout(0.1)(conv4)
#convolutional layers quantized the weights and the input  
conv4 = lq.layers.QuantConv2D(start_neuron *8,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(conv4)
conv4 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(conv4)
pool4 = tf.keras.layers.MaxPooling2D((2,2))(conv4)
pool4 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(pool4)

#conv4 = tf.keras.layers.Conv2D(start_neuron * 8, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(pool3)
#conv4 = tf.keras.layers.Dropout(0.2)(conv4)
#conv4 = tf.keras.layers.Conv2D(start_neuron * 8, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(conv4)
#pool4 = tf.keras.layers.MaxPooling2D((2,2))(conv4)

# middle
convmid = lq.layers.QuantConv2D(start_neuron * 16,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(pool4)
convmid = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(convmid)
convmid = tf.keras.layers.Dropout(0.3)(convmid)
convmid = lq.layers.QuantConv2D(start_neuron * 16,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(convmid)
convmid = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(convmid)

#convmid = tf.keras.layers.Conv2D(start_neuron * 16, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(pool4)
#convmid = tf.keras.layers.Dropout(0.3)(convmid)
#convmid = tf.keras.layers.Conv2D(start_neuron * 16, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(convmid)

#expansive path
deconv4 = lq.layers.QuantConv2DTranspose(start_neuron * 8, (2, 2), strides = (2,2),padding ='same',  **kwargs)(convmid)
uconv4 = tf.keras.layers.concatenate([deconv4,conv4])
uconv4 = lq.layers.QuantConv2D(start_neuron * 8,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(uconv4)
uconv4 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(uconv4)
uconv4 = tf.keras.layers.Dropout(0.2)(uconv4)
uconv4 = lq.layers.QuantConv2D(start_neuron * 8,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(uconv4)
uconv4 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(uconv4)

#deconv4 = tf.keras.layers.Conv2DTranspose(start_neuron * 8, (2, 2), strides = (2,2), padding = 'same')(convmid)
#uconv4 = tf.keras.layers.concatenate([deconv4,conv4])
#uconv4 = tf.keras.layers.Conv2D(start_neuron * 8, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(uconv4)
#uconv4 = tf.keras.layers.Dropout(0.2)(uconv4)
#uconv4 = tf.keras.layers.Conv2D(start_neuron * 8, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(uconv4)

deconv3 = lq.layers.QuantConv2DTranspose(start_neuron * 4, (2, 2), strides = (2,2),padding ='same',  **kwargs)(uconv4)
uconv3 = tf.keras.layers.concatenate([deconv3,conv3])
uconv3 = lq.layers.QuantConv2D(start_neuron * 4,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(uconv3)
uconv3 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(uconv3)
uconv3 = tf.keras.layers.Dropout(0.2)(uconv3)
uconv3 = lq.layers.QuantConv2D(start_neuron * 4,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(uconv3)
uconv3 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(uconv3)

#deconv3 = tf.keras.layers.Conv2DTranspose(start_neuron * 4, (2, 2), strides = (2,2), padding = 'same')(uconv4)
#uconv3 = tf.keras.layers.concatenate([deconv3,conv3])
#uconv3 = tf.keras.layers.Conv2D(start_neuron * 4, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(uconv3)
#uconv3 = tf.keras.layers.Dropout(0.2)(uconv3)
#uconv3 = tf.keras.layers.Conv2D(start_neuron * 4, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(uconv3)

deconv2 = lq.layers.QuantConv2DTranspose(start_neuron * 2, (2, 2), strides = (2,2),padding ='same',  **kwargs)(uconv3)
uconv2 = tf.keras.layers.concatenate([deconv2,conv2])
uconv2 = lq.layers.QuantConv2D(start_neuron * 2,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(uconv2)
uconv2 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(uconv2)
uconv2 = tf.keras.layers.Dropout(0.2)(uconv2)
uconv2 = lq.layers.QuantConv2D(start_neuron * 2,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(uconv2)
uconv2 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(uconv2)

#deconv2 = tf.keras.layers.Conv2DTranspose(start_neuron * 2, (2, 2), strides = (2,2), padding = 'same')(uconv3)
#uconv2 = tf.keras.layers.concatenate([deconv2,conv2])
#uconv2 = tf.keras.layers.Conv2D(start_neuron * 2, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(uconv2)
#uconv2 = tf.keras.layers.Dropout(0.1)(uconv2)
#uconv2 = tf.keras.layers.Conv2D(start_neuron * 2, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(uconv2)

deconv1 = lq.layers.QuantConv2DTranspose(start_neuron * 1, (2, 2), strides = (2,2),padding ='same',  **kwargs)(uconv2)
uconv1 = tf.keras.layers.concatenate([deconv1,conv1])
uconv1 = lq.layers.QuantConv2D(start_neuron * 1,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(uconv1)
uconv1 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(uconv1)
uconv1 = tf.keras.layers.Dropout(0.2)(uconv1)
uconv1 = lq.layers.QuantConv2D(start_neuron * 1,(3,3),**kwargs,activation = 'relu', padding ='same',pad_values=1.0)(uconv1)
uconv1 = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(uconv1)

#deconv1 = tf.keras.layers.Conv2DTranspose(start_neuron * 1, (2, 2), strides = (2,2), padding = 'same')(uconv2)
#uconv1 = tf.keras.layers.concatenate([deconv1,conv1])
#uconv1 = tf.keras.layers.Conv2D(start_neuron * 1, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(uconv1)
#uconv1 = tf.keras.layers.Dropout(0.1)(uconv1)
#uconv1 = tf.keras.layers.Conv2D(start_neuron * 1, (3,3),activation = 'relu',kernel_initializer ='he_normal', padding ='same')(uconv1)

#outputs = tf.keras.layers.Conv2D(1,(1,1), activation = 'sigmoid')(uconv1)
outputs = lq.layers.QuantConv2D(1,(1,1),**kwargs,activation = 'sigmoid')(uconv1)
outputs = tf.keras.layers.BatchNormalization(momentum = 0.9, scale = False, epsilon = 1e-4)(outputs)

def dice(y_true, y_pred):
        eps = 0.00001 
        y_true_f = tf.reshape(y_true,[tf.shape(y_true)[0],tf.shape(y_true)[1]*tf.shape(y_true)[2],tf.shape(y_true)[3]])
        y_pred_f = tf.reshape(y_pred,[tf.shape(y_pred)[0],tf.shape(y_pred)[1]*tf.shape(y_pred)[2],tf.shape(y_pred)[3]])                                                                                                
        intersection = eps + 2*tf.reduce_sum(y_true_f*y_pred_f, axis=1)                                                    
        # eps added in denomintor, to take care for DivisionByZero error.
        union = eps + tf.reduce_sum(y_true_f*y_true_f, axis=1) + tf.reduce_sum(y_pred_f*y_pred_f, axis=1)                                    
        IOU = intersection/union                                                                               
        return (tf.reduce_mean(IOU))
         
def dice_loss(y_true, y_pred):
        return -dice(y_true, y_pred)

def dice_metric(y_true, y_pred):
        y_true_f = K.cast(K.greater(y_true, 0.5), 'float32')
        y_pred_f = K.cast(K.greater(y_pred, 0.5), 'float32')
        
        return dice(y_true_f, y_pred_f)

model = tf.keras.Model(inputs =[inputs], outputs =[outputs])
#adam daptive Moment Estimation optimizer algorithm to update weights using adaptive learning rates
model.compile(optimizer ='adam', loss = 'binary_crossentropy' , metrics = ['accuracy',dice_metric])
model.summary()

#model checkpoint save the model after every epoch, save the best model only 
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_bnnpet.h5', verbose = 1, save_best_only = True )

logs = "logs/bnnunetpet" + datetime.now().strftime("%Y%m%d-%H%M%S")
#monitor the validation loss parameter, if the loss parameter does not get better then stop , 3 epochs further , if not improving stop
callbacks = [tf.keras.callbacks.EarlyStopping(patience = 30, monitor = 'val_loss'),tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')]
model.save('bnnunetpet.h5')

results = model.fit(XTrain, YTrain, validation_split = 0.1, batch_size = 16, epochs = 100, callbacks = callbacks)
lq.models.summary(model)
#######################

idx = random.randint(0, len(XTrain))

prediction_train = model.predict(XTrain[:int(XTrain.shape[0]*0.9)], verbose = 1) #training images
prediction_val = model.predict(XTrain[int(XTrain.shape[0]*0.9):], verbose = 1) #validation images
#prediction_test = model.predict(XTest, verbose = 1) #test images

prediction_train_t = (prediction_train > 0.5).astype(np.uint8)
prediction_val_t = (prediction_val > 0.5).astype(np.uint8)
#prediction_test_t = (prediction_test > 0.5).astype(np.uint8)
 
# perform sanity check on some random training samples
ix = random.randint(0, len(prediction_train_t))
imshow(XTrain[ix])
plt.show()
imshow(np.squeeze(prediction_train_t[ix]))
plt.show()

#perform sanity check on some random validation samples 
ix = random.randint(0, len(prediction_val_t))
imshow(XTrain[int(XTrain.shape[0]*0.9)][ix])
plt.show()
imshow(np.squeeze(YTrain[int(YTrain.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(prediction_val_t[ix]))
plt.show()

#plot the final validationi accuracy and loss 
plt.figure(1)
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend('train','test', loc = 'upper left')

print(np.max(results.history['accuracy']))
print(np.max(results.history['val_accuracy']))

plt.figure(2)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

print(np.min(results.history['loss']))
print(np.min(results.history['val_loss']))

plt.figure(3)
plt.plot(results.history['dice_metric'])
plt.plot(results.history['val_dice_metric'])
plt.title('dice metric')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

print(np.max(results.history['dice_metric']))