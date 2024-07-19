import tensorflow as tf
import os
import tensorflow_datasets as tfds

ds_name = 'imagenet2012'
[images, labels], info = tfds.load(ds_name, split='validation', as_supervised=True, with_info=True)
print(len(ds))

for i in range(5):
  print(labels[i])
