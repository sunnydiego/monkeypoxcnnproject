import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import sys
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report,f1_score,recall_score,precision_score,ConfusionMatrixDisplay,multilabel_confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.applications import ResNet50V2
from keras import applications
from keras.applications import ResNet50V2
from keras import Input
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation
from keras.layers.core import Flatten, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
#from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from matplotlib_inline import *
from sklearn.metrics import confusion_matrix
import itertools
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import matthews_corrcoef as MCC


#Path ve resize
datagen = ImageDataGenerator(rescale=1./255)
train_path = 'C:/Users/senko/Desktop/MonkeypoxProje/Fold1/Fold1/Fold1/Train'
val_path = 'C:/Users/senko/Desktop/MonkeypoxProje/Fold1/Fold1/Fold1/Val'
test_path = 'C:/Users/senko/Desktop/MonkeypoxProje/Fold1/Fold1/Fold1/Test'

IMG_SIZE = (120,120)

train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_path,
                                                                 image_size=IMG_SIZE,
                                                                 label_mode="categorical",
                                                                 batch_size=32)

test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_path,
                                                                 image_size=IMG_SIZE,
                                                                 label_mode="categorical",
                                                                 batch_size=32,
                                                                shuffle=False)

val_data = tf.keras.preprocessing.image_dataset_from_directory(directory=val_path,
                                                                 image_size=IMG_SIZE,
                                                                 label_mode="categorical",
                                                                 batch_size=32,
                                                                shuffle=False)





# Visualizing data
def visualize_random_images(dataset_type="train", label_type="Others"):

    sample = 9

    plt.figure(figsize=(15, 8))
    type_dir = train_path if dataset_type=="train" else test_path
    base_dir = os.path.join(type_dir, label_type)
    images = random.sample(os.listdir(base_dir), 9)

    for i, image in enumerate(images):
        plt.subplot(3, 3, i+1) 
        img = plt.imread(os.path.join(base_dir, image))
        plt.imshow(img)
visualize_random_images("train", "Monkeypox")



def show_confusion_matrix(history):
  
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()


# model1 tüm layerları dondur
# VGG16 pre-trained model
image_a, image_u = 120, 120
model1 = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (image_a, image_u, 3))
model1.summary()

#  layerları dondurma
for layer in model1.layers[:]:
    layer.trainable = False

# eğitilebilir layerlar
print("Trainable Layers:")
for i, layer in enumerate(model1.layers):
    print(i, layer.name, layer.trainable)

# yeni model üretmek için custom layerlar ekleme 
new_model = Sequential([
    model1,
    Flatten(name='flatten'),
    Dense(256, activation='relu', name='new_fc1'),
    Dropout(0.5),
    Dense(2, activation='softmax', name='new_predictions')
])
new_model.summary()


new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = new_model.fit(train_data,
                       epochs=5,
                       validation_data=val_data)



show_confusion_matrix(history)

new_model.evaluate(test_data)


y_pred = tf.math.round(new_model.predict(test_data))
y_true = []
for images, labels in test_data.unbatch():
  y_true.append(labels.numpy())
accuracy_score(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=train_data.class_names))


cm = multilabel_confusion_matrix(y_true, y_pred)
sns.heatmap(cm[0],annot=True,fmt='d')


#Model2  last layer unfreezeleme
#tamamen bağlantılı layerlar olmadan, farklı giriş boyutlarıyla VGG16 önceden eğitilmiş model 
image_a, image_u = 120, 120
model2 = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (image_a, image_u, 3))
#model2.summary()

# son layer haric hepsini dondurma
for i, layer in enumerate(model2.layers):
  layer.trainable = False
  if (i >=15 ):
    layer.trainable = True

for i, layer in enumerate(model2.layers):
    print(i, layer.name, layer.trainable)

# yeni model uretmek ıcın custom layer ekleme
new_model2 = Sequential([
    model2,
    Flatten(name='flatten'),
    Dense(512, activation='relu', name='new_fc1'),
    Dropout(0.5),
    Dense(2, activation='softmax', name='new_predictions')
])


#Model2  last layer unfreeze
# tamamen bağlantılı layerlar olmadan, farklı giriş boyutlarıyla VGG16 önceden eğitilmiş model 
image_a, image_u = 120, 120
model2 = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (image_a, image_u, 3))
#model2.summary()

#  son layer haric hepsini dondurma
for i, layer in enumerate(model2.layers):
  layer.trainable = True

for i, layer in enumerate(model2.layers):
    print(i, layer.name, layer.trainable)

# yeni model uretmek ıcın custom layer ekleme
new_model2 = Sequential([
    model2,
    Flatten(name='flatten'),
    Dense(512, activation='relu', name='new_fc1'),
    Dropout(0.5),
    Dense(2, activation='softmax', name='new_predictions')
])

# new_model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
#                 loss='binary_crossentropy',
#                 metrics=['accuracy'])

# history = new_model2.fit(train_data,
#                        epochs=5,
#                        validation_data=val_data)



# show_confusion_matrix(history)

# new_model2.evaluate(test_data)


# y_pred = tf.math.round(new_model2.predict(test_data))
# y_true = []
# for images, labels in test_data.unbatch():
#   y_true.append(labels.numpy())
# accuracy_score(y_true, y_pred)
# print(classification_report(y_true, y_pred, target_names=train_data.class_names))


# cm = multilabel_confusion_matrix(y_true, y_pred)
# sns.heatmap(cm[0],annot=True,fmt='d')




#cm = confusion_matrix(y_true, y_pred)
#sns.heatmap(cm.astype("int"), annot=True)

# y_true = train_batches.classes
# y_pred = new_model2.predict_generator(train_batches)
# y_pred = np.argmax(y_pred, axis=1)
# accuracy = accuracy_score(y_true, y_pred)
# print("accuracy Skoru: {}".format(accuracy))


# # Tahminleri yapma ve F1 skorunu hesaplama
# y_true = train_batches.classes
# y_pred = new_model2.predict_generator(train_batches)
# y_pred = np.argmax(y_pred, axis=1)
# f1score = f1_score(y_true, y_pred, average='weighted')
# recall = recall_score(y_true, y_pred, average='weighted')
# precision = precision_score(y_true, y_pred, average='weighted')
# print("F1 Skoru: {}".format(f1score))
# print("Recall Skoru: {}".format(recall))
# print("Precision Skoru: {}".format(precision))

# accuracy = (cm1[0][0] + cm1[1][1]) / np.sum(cm1)
# print("Accuracy: ", accuracy)



#ResNet50
image_a, image_u = 120, 120
model = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(image_a, image_u, 3))

model.summary()
    # yeni model uretmek ıcın custom layer ekleme
new_model2 = Sequential([
    model,
    Flatten(name='flatten'),
    Dense(512, activation='relu', name='new_fc1'),
    Dropout(0.5),
    Dense(2, activation='softmax', name='new_predictions')
])
new_model2.summary()


new_model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = new_model2.fit(train_data,
                       epochs=5,
                       validation_data=val_data)




show_confusion_matrix(history)

new_model2.evaluate(test_data)


y_pred = tf.math.round(new_model2.predict(test_data))
y_true = []
for images, labels in test_data.unbatch():
  y_true.append(labels.numpy())
accuracy_score(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=train_data.class_names))


cm = multilabel_confusion_matrix(y_true, y_pred)
sns.heatmap(cm[0],annot=True,fmt='d')





#DenseNet121
image_a, image_u = 120, 120
model = keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(image_a, image_u, 3))

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)
model.summary()
    # yeni model uretmek ıcın custom layer ekleme
new_model3 = Sequential([
    model,
    Flatten(name='flatten'),
    Dense(512, activation='relu', name='new_fc1'),
    Dropout(0.5),
    Dense(2, activation='softmax', name='new_predictions')
])
#new_model2.summary()

new_model3.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = new_model3.fit(train_data,
                       epochs=5,
                       validation_data=val_data)


show_confusion_matrix(history)

new_model3.evaluate(test_data)


y_pred = tf.math.round(new_model3.predict(test_data))
y_true = []
for images, labels in test_data.unbatch():
  y_true.append(labels.numpy())
accuracy_score(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=train_data.class_names))


cm = multilabel_confusion_matrix(y_true, y_pred)
sns.heatmap(cm[0],annot=True,fmt='d')



#EfficientNetB3
image_a, image_u = 120, 120
model = keras.applications.EfficientNetB3(include_top=False, weights='imagenet', input_shape=(image_a, image_u, 3))

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)
model.summary()
    # yeni model uretmek ıcın custom layer ekleme
new_model4 = Sequential([
    model,
    Flatten(name='flatten'),
    Dense(512, activation='relu', name='new_fc1'),
    Dropout(0.5),
    Dense(2, activation='softmax', name='new_predictions')
])
#new_model2.summary()
new_model4.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = new_model4.fit(train_data,
                       epochs=5,
                       validation_data=val_data)


show_confusion_matrix(history)

new_model4.evaluate(test_data)


y_pred = tf.math.round(new_model4.predict(test_data))
y_true = []
for images, labels in test_data.unbatch():
  y_true.append(labels.numpy())
accuracy_score(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=train_data.class_names))


cm = multilabel_confusion_matrix(y_true, y_pred)
sns.heatmap(cm[0],annot=True,fmt='d')

#Xception
image_a, image_u = 120, 120
model = keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(image_a, image_u, 3))

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)
model.summary()
    # yeni model uretmek ıcın custom layer ekleme 
new_model5 = Sequential([
    model,
    Flatten(name='flatten'),
    Dense(512, activation='relu', name='new_fc1'),
    Dropout(0.5),
    Dense(2, activation='softmax', name='new_predictions')
])
#new_model2.summary()
new_model5.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = new_model5.fit(train_data,
                       epochs=5,
                       validation_data=val_data)



show_confusion_matrix(history)

new_model5.evaluate(test_data)


y_pred = tf.math.round(new_model5.predict(test_data))
y_true = []
for images, labels in test_data.unbatch():
  y_true.append(labels.numpy())
accuracy_score(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=train_data.class_names))


cm = multilabel_confusion_matrix(y_true, y_pred)
sns.heatmap(cm[0],annot=True,fmt='d')



plt.plot(history.history['accuracy'],label="train accuracy")
plt.plot(history.history['val_accuracy'],label="val accuracy")
plt.title('model accuracy eval')
plt.style.use('fivethirtyeight')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'],loc='lower right')
plt.tight_layout
plt.show()




def plot_loss_curves(history):

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()



