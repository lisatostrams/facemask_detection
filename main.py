#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:44:44 2020

@author: lisatostrams
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from  tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory



#%%
path = 'data'
BATCH_SIZE = 32
init_epochs=10
fine_tuning_epochs=8
IMG_SIZE = (224, 224)
base_learning_rate=0.0001
output_path='output'



#%%
train_dataset = image_dataset_from_directory(path,
                                             shuffle=True,
                                             subset='training',
                                             seed=1,
                                             validation_split=0.2,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)
#%%
validation_dataset = image_dataset_from_directory(path,
                                             shuffle=True,
                                             subset='validation',
                                             seed=1,
                                             validation_split=0.2,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)


#%%


class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
#%%
    
    
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches//5)
validation_dataset = validation_dataset.skip(val_batches//5)

#%%


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%


data_augmentation=tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(.5, .2),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
])


#%%


image, label = next(iter(validation_dataset))


#%%



plt.figure(figsize=(3,3))
plt.imshow(image[0].numpy().astype('uint8'))
plt.title("original image")
plt.axis("off")
plt.show()
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    data_aug_image=data_augmentation(tf.expand_dims(image[0],0))
    plt.imshow(data_aug_image[0]/255.0)
    plt.title("augumentation image")
    plt.axis("off")
plt.show()




#%%


image_shape=IMG_SIZE+(3,)
base_model=tf.keras.applications.EfficientNetB0(input_shape=image_shape,
                                    include_top=False,
                                    weights='imagenet',
                                    drop_connect_rate=0.4)

#%%

model=tf.keras.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1))

model.summary()

#%%

input=tf.keras.Input(image_shape)
x=data_augmentation(input)
x=base_model(x,training=False)
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.BatchNormalization()(x)
output=tf.keras.layers.Dense(1)(x)
model=tf.keras.Model(input,output)

model.summary()

#%%


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


#%%
print('callbacks')
checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

csv_logger = tf.keras.callbacks.CSVLogger('output/training.log')


history=model.fit(train_dataset,
          epochs=init_epochs,
          validation_data=validation_dataset,
          callback = [cp_callback,csv_logger]
         )

#%%



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,0.2])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.save_fig('output/results.png',dpi=300)


