import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

fpath = tf.keras.utils.get_file(origin='https://images-ng.pixai.art/images/orig/b483ad6b-d2f8-4b57-8c3c-b90bfc804806', cache_dir='/content')
print(str(fpath))
img = tf.keras.utils.load_img(str(fpath))
plt.imshow(img)

array = tf.keras.utils.img_to_array(img)


array = tf.keras.utils.img_to_array(img)
array[:,:,0] = np.zeros_like(array[:,:,-1])
array[:,:,1] = np.zeros_like(array[:,:,-1])
new_array = tf.cast(array, dtype='int8')

plt.imshow(new_array) # 0, 1, 2

r = array[:,:,0]
g = array[:,:,1]
b = array[:,:,2]

img = tf.stack([r,g,b], axis=-1)
img/=255.0
plt.imshow(img)

import tensorflow.keras.layers as lay

# The first neural network we made (yes, this is it!)
# We will get into the specifics of how this works next session

model = tf.keras.models.Sequential([
    lay.Flatten(input_shape=(92,92,3)),
    lay.Dense(5),
    lay.Dense(5),
    lay.Dense(5),
    lay.Dense(5),
    lay.Dense(3)
])

model.summary()