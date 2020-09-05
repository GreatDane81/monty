import generate_train_data

from tensorflow import keras 


import numpy as np

# for conv:
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Flatten, Conv2D

from Game import Game
import Board

from tensorflow.keras import backend as K

# for chess
import chess
from chess.engine import Cp, Mate, MateGiven
import chess.pgn

# for model storage
from tensorflow.keras.models import load_model

batch_size = 128

np_board_shape = (9, 8, 13)
np_board_shape_flat = 9*8*13

#model = keras.Sequential([
#    keras.layers.InputLayer(input_shape=(936,)),
#    keras.layers.Dense(512, activation='relu'),
#    keras.layers.Dense(512, activation='relu'),
#    keras.layers.Dense(1)])

conv_model = Sequential()
inp = keras.layers.Input(batch_input_shape=(None,9,8,13))# ok just assume 13 channels for now
conv_model.add(inp)
conv_model.add(Conv2D(256, kernel_size=3, activation='relu')) # TODO: the one here is usually for grayscale images, not sure how this will work with 1/0 bin
conv_model.add(Conv2D(256, kernel_size=3, activation='relu')) # kernel size 4 works, 5 does not because of dim input (none, 9, 8, 13), idk?
conv_model.add(Flatten())
conv_model.add(Dense(1))


optimizer = keras.optimizers.Adam(lr=0.1)
conv_model.compile(optimizer=optimizer,loss='mean_absolute_error') 
print("compiled successfully")



# getting the training data
out_file_path = "C:/Users/Ethan/Documents/GitHub/monty/training_out_file"
train_list = generate_train_data.load_training_data(out_file_path)

positions = []
scores = []
for example in train_list:
    # get the board, score
    board, score = example[0], example[1]
    board = np.array(board) # TODO: .flatten() for the non-convolutional model
    positions.append(board)
    scores.append(score)

print(positions[0].shape, type(positions[10]))
print(type(scores[0]))

scores_sum = np.sum(scores)
scores_std = np.std(scores)
print(scores_sum, "score sum", scores_std, "score std")

#for i in range(0, len(scores)):
#    scores[i] = (scores[i] - scores_sum)/scores_std

partial_train_index = len(positions)//2 + len(positions)//4 # use 75% of training data for fitting, 25% for validation
training_positions = np.array(positions[:partial_train_index]) # convert to NP to make it readable for keras, trying to flatten
validation_positions = np.array(positions[partial_train_index:])
training_scores = np.array(scores[:partial_train_index])
validation_scores = np.array(scores[partial_train_index:]) # the tale of a misplaced colon costing me an hour

print(training_positions.shape, "tp shape", type(training_positions)) # (100, 936)
print(training_scores.shape, "ts shape", type(training_scores)) # (100,) So this should work, but for some reason it's complaining x doesn't fit y.
print(training_positions[0].shape, "one example shape")


print(conv_model.summary())


history = conv_model.fit(training_positions,
                        training_scores,
                        epochs=5,
                        batch_size=128,
                        validation_data=(validation_positions,validation_scores),
                        shuffle=True)

#conv_model.save('conv_model.h5') # saves the whole file into this file




train_path_tal = "C:/Users/Ethan/Documents/GitHub/monty/Tal.pgn"
tal_file = open(train_path_tal)

game = chess.pgn.read_game(tal_file)

np_board =  Board.Board()
py_board = chess.Board()
print(np.array(np_board.board).shape)


SF_DEPTH = 10 # depth for stockfish

ANALYSIS_TIME = 0.1 # in seconds

path = "C:/Users/Ethan/Documents/GitHub/monty/stockfish/stockfish-11-win/stockfish-11-win/Windows/stockfish_20011801_x64"

engine = chess.engine.SimpleEngine.popen_uci(path)

for move in game.mainline_moves():
    # update the np board
    Game.play_move_on_np_board(np_board, move)
    py_board.push(move)
    score =  engine.analyse(py_board, chess.engine.Limit(time=ANALYSIS_TIME))["score"]
    my_board = np.array(np_board.board)
    prediction = conv_model.predict(np.array([my_board,])) # Ok so it was expecting a list of predictions, for a single prediction use this
    print("score:",score,"prediction:",prediction)
    #numerical_score = get_numerical_score(score)