import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

data = tfds.load('cifar10', split='train+test', shuffle_files=True, as_supervised=True)

# split data into train and test
train_data = data.take(50000)
val_data = data.skip(50000)

# convert to numpy arrays
train_data = tfds.as_numpy(train_data)
val_data = tfds.as_numpy(val_data)

# get the data and labels
train_data = np.array([x[0] for x in train_data])
train_labels = np.array([x[1] for x in train_data])

val_data = np.array([x[0] for x in val_data])
val_labels = np.array([x[1] for x in val_data])

