import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split



def transform_labels(labels):
    new_labels = np.zeros((len(labels), 10))
    for i in range(0, len(labels)):
        label = int(labels[i]-1)
        new_labels[i][label] = 1
    return new_labels

#%%
start = time.time()
images = []
directory = 'train/train/'

for label in range(1,11):
        data =  np.stack([np.array(Image.open(directory+str(label)+"/"+image)) 
                for image in os.listdir(directory+str(label))], axis=0)
        images.append(data)
images =  np.concatenate(images)
#%%
import matplotlib.pyplot as plt
plt.imshow(images[0], cmap='gray')
plt.show()

np.save('training_data', images)
#%%
training_data = np.load(r'C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment1/training_data.npy').astype("float32")
#%%
# Muodosta harjoitusjoukko ja sille label-joukko
trainX = np.expand_dims(training_data / 255.0, axis=3)
trainY = np.ones(trainX.shape[0]).astype("int32")
for n in range(1,11):
    trainY[(n-1)*6000 : n*6000] *= n

trainY_input = transform_labels(trainY).astype("int32")

trainX, validX, trainY_input, validY_input = train_test_split(trainX, 
                                                              trainY_input, 
                                                              test_size=0.2,
                                                              random_state=1)
#%% Base model
dropout = 0.1
starting_filters = 16 
kernel = (3, 3)
model = Sequential()

model.add(Conv2D(starting_filters, kernel_size=kernel, activation='relu', 
                 kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(starting_filters*4, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
#%% Base model
opt = tf.optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 10
history = model.fit(trainX, trainY_input, epochs=epochs, 
                    validation_data=(validX, validY_input), verbose=2)

#Plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%% Complex model
dropout = 0.2
starting_filters = 8 
kernel = (3, 3)
model = Sequential()

model.add(Conv2D(starting_filters, kernel_size=kernel, activation='relu', 
                 padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(starting_filters, kernel_size=kernel, activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.55))

model.add(Conv2D(2*starting_filters, kernel_size=kernel, activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(2*starting_filters, kernel_size=kernel, activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.7))

model.add(Conv2D(4*starting_filters, kernel_size=kernel, activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(4*starting_filters, kernel_size=kernel, activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.85))

model.add(Flatten())
model.add(Dense(4*starting_filters, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout))
model.add(Dense(10, activation='softmax'))


#%% Fitting
epochs = 50
batch_size = 128
learning_rate = 0.01
print(f"Running {epochs} epochs with batch size of {batch_size} \
      and learning rate of {learning_rate}.")
opt = tf.optimizers.Adam(lr=learning_rate)
callback = EarlyStopping(monitor='val_accuracy', patience=10, 
                         restore_best_weights=True, mode="max")
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainX, trainY_input, epochs=epochs, 
                    validation_data=(validX, validY_input), verbose=2,
                    batch_size=batch_size, use_multiprocessing=True, 
                    workers=16, callbacks=[callback])
'''
#Prediction
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = class_acc(y_pred, Y_test)
print(f"Accuracy: {accuracy}")'''


#Plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'test'], loc='upper left')
plt.show()


