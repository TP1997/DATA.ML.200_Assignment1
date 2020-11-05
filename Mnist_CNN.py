import numpy as np

# Itse käytän tätä koska Linux.
root = '/home/tuomas/Python/DATA.ML.200/DATA.ML.200_Assignment1/'
training_data = np.load(root + 'training_data.npy')

#%%
# Muokkaa & käytä tätä jos Windows masiina.
training_data = np.load('training_data.npy')

#%%
# Muodosta harjoitusjoukko ja sille label-joukko
trainX = training_data / 255
trainY = np.ones(trainX.shape[0])
for n in range(1,10):
    trainY[(n-1)*6000 : n*6000] *= n
    


#%%
import matplotlib.pyplot as plt
img = training_data[6000]
plt.imshow(img, cmap='gray')
plt.show()