#Importit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GaussianDropout, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
                                                              test_size=0.3,
                                                              random_state=1)
# Luodaan vielä erillinen testijoukko
validX, testX, validY_input, testY_input = train_test_split(validX,
                                                            validY_input,
                                                            test_size=0.1,
                                                            random_state=1)

#%% Complex model 
# Överi. Hieman muokattu versio introkurssin cifar10-verkosta. 
# Saavuttaa ~99% validation accuracyn. Ei cross-validaatiota, joten mahd. overfittaa.
dropout = 0.2
starting_filters = 8 
kernel = (3,3)
model = Sequential()

model.add(Conv2D(starting_filters, kernel_size=kernel, activation='relu', padding="same", kernel_initializer='he_uniform', input_shape = (28, 28, 1)))
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
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout))
model.add(Dense(10, activation='softmax'))

#%% Model 2
dropout = 0.2
starting_filters = 16 
kernel = (3,3)
model = Sequential()

model.add(Conv2D(starting_filters, kernel_size=kernel, activation='relu', padding="same", kernel_initializer='he_uniform', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(starting_filters, kernel_size=kernel, activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(starting_filters, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.55))

model.add(Conv2D(2*starting_filters, kernel_size=kernel, activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(2*starting_filters, kernel_size=kernel, activation='relu', padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(2*starting_filters, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.7))

model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout))
model.add(Dense(10, activation='softmax'))
#%% Model 3
model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

#%% Model 4 
dropout = 0.17
starting_filters = 16 
kernel = (3, 3)
model = Sequential()

model.add(Conv2D(starting_filters, kernel_size=kernel, activation='relu', 
                 padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.4))

model.add(Conv2D(2*starting_filters, kernel_size=kernel, activation='relu',
                 padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.6))

model.add(Conv2D(4*starting_filters, kernel_size=kernel, activation='relu', 
                 padding="same", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout * 0.8))

model.add(Flatten())
model.add(Dense(8*starting_filters, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(GaussianDropout(dropout))
model.add(Dense(10, activation='softmax'))

#%% Fitting
epochs = 100
batch_size = 128
learning_rate = 0.015
print(f"Running {epochs} epochs with batch size of {batch_size} and learning rate of {learning_rate}.")
opt = tf.optimizers.Adam(lr=learning_rate)

# Keskeyttää fitin, jos validation accuracy ei parane 10 epochiin.
callback = EarlyStopping(monitor='val_accuracy', patience=20, 
                         restore_best_weights=True, mode="max")
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#%% With data augmentation
idg = ImageDataGenerator(width_shift_range=0.1, 
                         height_shift_range=0.1, 
                         horizontal_flip=False,
                         rotation_range=10,
                         zoom_range=0.1,
                         fill_mode='nearest')
itd = idg.flow(trainX, trainY_input, batch_size=batch_size)

history = model.fit(itd, 
                    steps_per_epoch=int(trainX.shape[0] / batch_size),
                    epochs=epochs, 
                    validation_data=(validX, validY_input), 
                    batch_size=batch_size, 
                    #use_multiprocessing=True, 
                    #workers=16, 
                    callbacks=[callback],
                    verbose=1)
#%% Without data augmentation
history = model.fit(trainX, trainY_input, 
                    epochs=epochs, 
                    validation_data=(validX, validY_input), 
                    batch_size=batch_size, 
                    use_multiprocessing=True, 
                    workers=16, 
                    callbacks=[callback],
                    verbose=1)
'''
#Prediction
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = class_acc(y_pred, Y_test)
print(f"Accuracy: {accuracy}")'''

#%% Plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%% Plotting 2

plt.figure(figsize=(10,10))
# Plot model's accuracy information based on epochs
plt.subplot(211)
plt.title('Accuracy evolution')
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1.0])
plt.legend(loc = 'lower right')

# Plot model's loss function information based on epochs
plt.subplot(212)
plt.title('Loss function evolution')
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc = 'lower right')

#%% Evaluoi lopullinen tarkkuus
test_loss, test_acc = model.evaluate(testX, testY_input, verbose=2)
print("Accuracy using model.evaluate: {}".format(test_acc))

#%% Evaluoi lopullinen tarkkuus 2
predY = np.argmax(model.predict(testX, verbose=2), axis=1)
testY = np.argmax(testY_input, axis=1)
acc = np.sum(predY == testY) / testY.shape[0]
print("Accuracy using model.predict: {}".format(acc))

#%% Mallin tallennus
model.save("saved_models/<MALLIN NIMI>")
#%% Mallin ja testidatan nouto (windows)
model = tf.keras.models.load_model("C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/saved_models/cnn_model - 70-30 - valAcc0.994")

win_root = r'C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment1/'
test_data = np.load(win_root + 'test_data.npy').astype("float32") / 255.0
test_data = np.expand_dims(test_data, axis=3)
#%% Testidatan muokkaus ja ennustus
y_pred = np.argmax(model.predict(test_data), axis=1)
pred_labels = y_pred+np.ones(y_pred.shape, dtype=int) 
#%% kilpailutiedoston muodostaminen
with open("sample_submission.csv", "w") as fp: 
    fp.write("Id,Category\n") 
    for idx in range(10000): 
        fp.write(f"{idx:05},{pred_labels[idx]}\n") 
