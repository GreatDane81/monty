import generate_train_data

from tensorflow import keras

import matplotlib.pyplot as plt

import numpy as np

batch_size = 128

np_board_shape = (9, 8, 13)
np_board_shape_flat = 9*8*13

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=np_board_shape_flat),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)])



model.compile(optimizer='rmsprop',loss='mean_squared_error')


# getting the training data
out_file_path = "C:/Users/Ethan Dain/Desktop/University/Machine Learning/Code/monty/training_out_file"
train_list = generate_train_data.load_training_data(out_file_path)

positions = []
scores = []
for example in train_list:
    # get the board, score
    board, score = example[0], example[1]
    flat_board = np.array(board).flatten()
    positions.append(flat_board)
    scores.append(score)

print(positions[0].shape, type(positions[10])) # so successfully flattened
print(type(scores[0]))

partial_train_index = 50
training_positions = np.array(positions[:partial_train_index]) # convert to NP to make it readable for keras, trying to flatten
validation_positions = np.array(positions[partial_train_index:])
training_scores = np.array(scores[:partial_train_index])
validation_scores = np.array(scores[:partial_train_index])

print(training_positions.shape, "tp shape") # (100, 936)
print(training_scores.shape, "ts shape") # (100,) So this should work, but for some reason it's complaining x doesn't fit y.


print(model.summary())

history = model.fit(training_positions, 
                    training_scores,
                    epochs = 5,
                    batch_size=partial_train_index,
                    validation_data=(validation_positions, validation_scores))