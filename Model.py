# My files

# for loading the training data
import generate_train_data

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

# for analsyis of distribution of scores
import pandas as pd

# for data sorting
from operator import itemgetter

conv_model = Sequential()
inp = keras.layers.Input(batch_input_shape=(None,9,8))# ok just assume 13 channels for now
conv_model.add(inp)
conv_model.add(Dense(128, activation='relu')) # TODO: the one here is usually for grayscale images, not sure how this will work with 1/0 bin
conv_model.add(Dropout(rate=0.2))
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
    board = np.array(board)
    positions.append(board)
    scores.append(score)



print(conv_model.summary())

history = conv_model.fit(positions,
                        scores,
                        epochs=2,
                        batch_size=64,
                        validation_split=0.2,
                        shuffle=True)

conv_model.save('conv_model.h5') # saves the whole model into this file
