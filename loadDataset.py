import tensorflow as tf
import os
import tensorflow_datasets as tfds

ds_name = 'imagenet2012'
ds, info = tfds.load(ds_name, split='validation', as_supervised=True, with_info=True, data_dir='~/tensorflow_datasets/downloads/manual/imagenet2012')
print(len(ds))
