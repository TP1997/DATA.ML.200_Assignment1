Ajettu Google Colabissa, joten jos TF-versio on alle 2.3, ei välttämättä aukea.
Käytetty idg:
#%% With data augmentation
idg = ImageDataGenerator(width_shift_range=0.1, 
                         height_shift_range=0.1, 
                         horizontal_flip=False,
                         rotation_range=1,
                         zoom_range=0.1,
                         fill_mode='nearest')