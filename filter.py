import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

num_skip = 0
for folder in ('Cat', 'Dog'):
    folder_path = os.path.join("PetImages", folder)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, 'rb')
            is_jfif = tf.compat.as_bytes('JFIF') in fobj.peek(10)
        finally:
            fobj.close()
            
        if not is_jfif:
            num_skip += 1
            os.remove(fpath)

img_size = (180, 180)
b_size = 128

train_ds, vall_ds = tf.keras.utils.image_dataset_from_directory(
    'PetImages',
    validation_split=0.2,
    subset='both',
    seed=1337,
    image_size=img_size,
    batch_size=b_size,
)

plt.figure(figsize=(10,10))
for img, label in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.axis('off')
        if label[i] == 1:
            plt.title('ipen')
        else:
           plt.title('cat')
            
plt.show()

        
