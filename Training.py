import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplim
import zipfile
import os
import random 
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import VGG16
import PIL
import cv2
import pathlib
import pickle

#Training and test directories
train_dir = f"{os.getcwd()}/chest_xray/train/"
test_dir = f"{os.getcwd()}/chest_xray/test/"

## Initialize preprocessing conditions
#VGG16 is trained on coloured images with 3 chanels
#Therefore we need to convert our greyscale images to 
#pseudocolored images with 3 chanels (repeat the same image in all 3 chanels)
#Thats why we use color_mode = "rgb". Keras will automatically
#convert the images to rgb format
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    zoom_range = 0.2,
    rotation_range = 0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode = "rgb",
    target_size = (150, 150),
    batch_size = 75,
    class_mode = "binary"
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    color_mode = "rgb",
    target_size = (150, 150),
    batch_size = 75,
    class_mode = "binary"
)

#Initialize the base model
#Use weights trained with Imagenet
vgg = VGG16(input_shape = (150, 150, 3), 
            include_top = False, 
            weights = "imagenet")

#Freeze all of its layers
vgg.trainable = False

#Add new model on top
inputs = tf.keras.Input(shape = (150, 150, 3))
x = vgg(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(256,kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)                  
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)           

model = tf.keras.Model(inputs, outputs) 

model.compile(optimizer = SGD(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.AUC()])

## Early stopping callback : monitoring validation loss
callback_early = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    min_delta = 0.01,
    patience = 5
)

#Training the layers on top until convergence
#(Until the validation loss stops decreasing)
history = model.fit(
    train_generator,
    validation_data = test_generator,
    steps_per_epoch = 50,
    epochs = 50,
    validation_steps = 8,
    verbose = 1,
    callbacks = [callback_early]
)

#Save history
with open( f"{os.getcwd()}vgg_history", "wb") as file_pi:
        pickle.dump(history.history, file_pi)

#Plot accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.subplot(1,2,1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

#Plot AUC and precision
pre = history.history['precision_2']
val_pre = history.history['val_precision_2']
auc = history.history['auc_2']
val_auc = history.history['val_auc_2']

plt.subplot(1,2,1)
plt.plot(epochs, pre, 'r', label='Training precision')
plt.plot(epochs, val_pre, 'b', label='Validation precision')
plt.title('Training and validation precision')
plt.legend(loc=0)
plt.figure()
plt.show()

plt.subplot(1,2,2)
plt.plot(epochs, auc, 'r', label='Training AUC')
plt.plot(epochs, val_auc, 'b', label='Validation AUC')
plt.title('Training and validation AUC')
plt.legend(loc=0)
plt.figure()
plt.show()

#Fine tune the whole model to improve performance
## Early stopping callback : monitoring validation loss
#Decrease patience because fine tuning can easily lead to overfitting
callback_finetune = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    min_delta = 0.01,
    patience = 2
)

#Unfreeze the base model
vgg.trainable = True

#Fine tune the last block of the vgg network
for layer in vgg.layers[:15]:
  layer.trainable = False

#Compile the model again so that the changes will take effect
model.compile(optimizer = SGD(lr=0.0001, momentum = 0.9), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.AUC()])

#Set the learning rate to exponentially decrease with every epoch
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

#Fine tuning
history_fine = model.fit(
    train_generator,
    validation_data = test_generator,
    steps_per_epoch = 50,
    epochs = 20,
    validation_steps = 8,
    verbose = 1,
    callbacks = [callback_finetune, lr_schedule]
)

#Save model weights and history
model.save( f"{os.getcwd()}/vgg16_model")

with open( f"{os.getcwd()}/vgg_fine_tune_history", "wb") as file_pi:
        pickle.dump(history_fine.history, file_pi)

#Load history files
vgg_history = pickle.load(open(f"{os.getcwd()}/vgg_history", "rb"))
vgg_fine_tune_history = pickle.load(open(f"{os.getcwd()}/vgg_fine_tune_history", "rb"))

#Merge the two dictionaries
vgg_history["accuracy"].extend(vgg_fine_tune_history["accuracy"])
vgg_history["val_accuracy"].extend(vgg_fine_tune_history["val_accuracy"])
vgg_history["loss"].extend(vgg_fine_tune_history["loss"])
vgg_history["val_loss"].extend(vgg_fine_tune_history["val_loss"])

#Plot accuracy and loss
acc_full = vgg_history['accuracy']
val_acc_full = vgg_history['val_accuracy']
loss_full = vgg_history['loss']
val_loss_full = vgg_history['val_loss']

epochs = range(len(acc_full))

plt.subplot(1,2,1)
plt.plot(epochs, acc_full, 'r', label='Training accuracy')
plt.plot(epochs, val_acc_full, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)

plt.subplot(1,2,2)
plt.plot(epochs, loss_full, 'r', label='Training loss')
plt.plot(epochs, val_loss_full, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.savefig(f"{os.getcwd()}/Final_Acc_Loss.png")