#Importit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


#%%
# Itse käytän tätä koska Linux.
root = '/home/tuomas/Python/DATA.ML.200/DATA.ML.200_Assignment1/'
training_data = np.load(root + 'training_data.npy')

#%%
# Muokkaa & käytä tätä jos Windows masiina.
win_root = r'C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment1/'
training_data = np.load(win_root + 'training_data.npy')

#%%
# Muodosta harjoitusjoukko ja sille label-joukko
trainX = training_data.astype("float32") / 255.0 #normalisointi
# Lisää yksi ulottuvuus, jotta neuroverkko osaa vastaanottaa datan
# (60000, 28, 28) -> (60000, 28, 28, 1)
trainX = np.expand_dims(trainX, axis=3)
trainY = np.ones(trainX.shape[0]).astype("int32")
for n in range(1,11):
    trainY[(n-1)*6000 : n*6000] *= n

# Muokkaa labelit "one-hot encoding" muotoon neuroverkkoa varten
trainY_input = np.zeros((len(trainY), 10)).astype("int32")
for i in range(0, len(trainY)):
    label = int(trainY[i]-1) # Vähennä 1, koska datan numerot 1-10 labeloidaan välille 0-9.
    trainY_input[i][label] = 1

# Harjoitus-validointi split
trainX, validX, trainY_input, validY_input = train_test_split(trainX, 
                                                              trainY_input, 
                                                              test_size=0.2,
                                                              random_state=1)

#%%
import matplotlib.pyplot as plt
img = training_data[6000]
plt.imshow(img, cmap='gray')
plt.show()

#%% Complex model 
# Överi. Hieman muokattu versio introkurssin cifar10-verkosta. 
# Saavuttaa ~99% validation accuracyn. Ei cross-validaatiota, joten mahd. overfittaa.
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
# Keskeyttää fitin, jos validation accuracy ei parane 10 epochiin.
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