
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import random
import cv2
import os
from keras.preprocessing.image import load_img


image_dir='images'
model = keras.models.load_model('model/generator.h5')
print(model.summary())
inputs = keras.Input((None, None, 4))
outputs = model(inputs)
model = keras.models.Model(inputs, outputs)



# Surrogate file reader:
image_paths =[]

for root, dirs, files in os.walk(os.path.abspath("images/")):
    for file in files:
        image_paths.append(os.path.join(root, file))


print("---Image Path---")
print(image_paths)
print("---End--")


# Select images to plot, max 6
n_images = len(image_paths)

for idx in range(n_images):
    fig, ax = plt.subplots(1, 6, figsize=(30, 20))
    for col in range(6):
        image = cv2.imread(image_paths[idx], 1)

        #cv2.imshow('a',image)
        #cv2.waitKey()
        image = cv2.resize(image, (128, 128)) #input size :((
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = (image / 255) * 2 - 1

        
        h, w, _ = image.shape
        if col %2== 0:
            ax[col].imshow((image + 1.0) / 2.0)
            ax[col].set_title('Original')
        else:
            condition = np.ones(shape=[1, h, w, 1]) * (col - 1)
            conditioned_images = np.concatenate([np.expand_dims(image, axis=0), condition], axis=-1)
            aged_image = model.predict(conditioned_images)[0]
            ax[col].imshow((aged_image + 1.0) / 2.0)
            ax[col].set_title('Old ' + str(col))

    plt.subplots_adjust(wspace=0.1,left=0.03,right=0.97,top=0.99)
    plt.show()
    cv2.waitKey()
