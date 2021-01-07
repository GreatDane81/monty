# My files

# for loading the training data
import generate_train_data
import BoardTensor

# for paths
import paths


# Other libraries/files

import numpy as np

# for conv_model:
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

from tensorflow.keras import backend as K

# for chess
import chess
from chess.engine import Cp, Mate, MateGiven
import chess.pgn

# for model storage
from tensorflow.keras.models import load_model


# for data sorting
from operator import itemgetter

# The model architecture was copied from:
# http://www.diva-portal.se/smash/get/diva2:1366229/FULLTEXT01.pdf
#


# Doing some wizardry from SO to solve a tensorflow problem
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

conv_model = Sequential()
inp = keras.layers.Input(batch_input_shape=(None,8,8,7))# ok just assume 13 channels for now
conv_model.add(inp)
conv_model.add(Conv2D(8, kernel_size=3, activation='relu')) # TODO: the one here is usually for grayscale images, not sure how this will work with 1/0 bin
conv_model.add(Conv2D(16, kernel_size=3, activation='relu')) # TODO: the one here is usually for grayscale images, not sure how this will work with 1/0 bin
conv_model.add(Conv2D(32, kernel_size=3, activation='relu')) # TODO: the one here is usually for grayscale images, not sure how this will work with 1/0 bin
conv_model.add(Conv2D(64, kernel_size=2, activation='relu')) # TODO: the one here is usually for grayscale images, not sure how this will work with 1/0 bin
conv_model.add(Dense(64))
conv_model.add(Flatten())
conv_model.add(Dense(1))

optimizer = keras.optimizers.Adam()

# I did a lot of research on this in between commits. Just keep it as mse for now, it's the defacto loss function.
# Low loss means nothing, because it will just guess an even score, as that's what most positions are. Point is you need to identify winning
# features, and mse will punish the model for missing.
conv_model.compile(optimizer=optimizer,loss='mse')
print("compiled successfully")


# getting the training data
train_list = generate_train_data.load_training_data(paths.OUTFILE_PATH)


# now extract the training data
positions = []
scores = []
for example in train_list:
    # get the board, score
    board, score = example[0], example[1] 
    positions.append(board)
    scores.append(score)

positions = np.asarray(positions)
scores = np.asarray(scores).astype('float32')


print(conv_model.summary())
history = conv_model.fit(positions,
                        scores,
                        epochs=2,
                        batch_size=64,
                        validation_split=0.2,
                        shuffle=True)

conv_model.save('conv_model.h5') # saves the whole model into this file
