import tensorflow as tf
import os
import tensorflow_datasets as tfds

ds_name = 'imagenet2012'
ds, info = tfds.load(ds_name, split='validation', as_supervised=True, with_info=True)
print(len(ds))

for image, label in ds:
  print(label)
  break
