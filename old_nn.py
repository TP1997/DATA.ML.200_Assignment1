import numpy as np
import time
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image


def class_acc(pred, gt):
    return np.average(pred == gt)


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


def transform_labels(labels):
    new_labels = np.zeros((len(labels), 10))
    for i in range(0, len(labels)):
        label = labels[i]
        new_labels[i][label] = 1
    return new_labels


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
training_data = [
X_train = (np.concatenate([d["data"] for d in training_data], axis=0).astype("float32")) / 255
Y_train = np.concatenate([d["labels"] for d in training_data], axis=0)
X_train = X_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
Y_train_input = transform_labels(Y_train)
test_data = unpickle('cifar-10-batches-py/test_batch')
X_test = (test_data["data"].astype("float32")) / 255
Y_test = np.array(test_data["labels"])
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
end = time.time()
print(f"Time for loading and preprocessing the data: {(end - start):.2f}s")


#%%


#Complex model
dropout = 0.7
model = Sequential()

model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.55))

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.7))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.85))

model.add(Flatten())
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout))
model.add(Dense(10, activation='softmax'))


# Fitting
epochs = 10
batch_size = 128
learning_rate = 0.1
print(f"Running {epochs} epochs with batch size of {batch_size} and learning rate of {learning_rate}.")
tf.optimizers.Adam(lr=learning_rate)
callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode="max")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train_input, epochs=epochs, validation_split=0.2, verbose=2,
                    batch_size=batch_size, use_multiprocessing=True, workers=16, callbacks=[callback])

#Prediction
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = class_acc(y_pred, Y_test)
print(f"Accuracy: {accuracy}")


#Plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'test'], loc='upper left')
plt.show()


