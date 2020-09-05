import generate_train_data

from tensorflow import keras 


import numpy as np

# for conv_model:
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

# for analsyis of distribution of scores
import pandas as pd


# for data sorting
from operator import itemgetter


np_board_shape = (9, 8, 13)
np_board_shape_flat = 9*8*13


conv_model = Sequential()
inp = keras.layers.Input(batch_input_shape=(None,9,8,13))# ok just assume 13 channels for now
conv_model.add(inp)
conv_model.add(Conv2D(256, kernel_size=3, activation='selu')) # TODO: the one here is usually for grayscale images, not sure how this will work with 1/0 bin
conv_model.add(Flatten())
conv_model.add(Dense(1))

optimizer = keras.optimizers.Adam()

conv_model.compile(optimizer=optimizer,loss='mean_absolute_error')  # maybe the idea should be use functions that identify "average case" which are by far
# more common that outlier "totally winning" situations to fit?
print("compiled successfully")



# getting the training data
out_file_path = "C:/Users/Ethan/Documents/GitHub/monty/training_out_file"
train_list = generate_train_data.load_training_data(out_file_path)

# going to try sorting by scores, and select data by each quintile in equal proportions

sorted_train_list = sorted(train_list,key=itemgetter(1))
print(sorted_train_list[0][1]) # nice ok, so now everything is sorted.
# might want to do this by absolute value, just so "big scores" are stored.

sorted_positions = []
sorted_scores = []
for example in sorted_train_list:
    board, score = example[0], example[1]
    board = np.array(board)
    sorted_positions.append(board)
    sorted_scores.append(score)
# going to try something a bit ad hoc, but: take bottom 10%, middle 10%, top 10% and try training on that. see what the results are


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
scores_mean = np.mean(scores)
scores_std = np.std(scores)
print(scores_sum, "score sum", scores_std, "score std")

#standardize data to make training the model easier 
#for i in range(0, len(scores)):
#    scores[i] = (scores[i] - scores_mean)/scores_std

partial_train_index = len(positions)//2 + len(positions)//4 # use 75% of training data for fitting, 25% for validation
#training_positions = np.array(positions[:partial_train_index]) # convert to NP to make it readable for keras, trying to flatten
validation_positions = np.array(positions[partial_train_index:])
#training_scores = np.array(scores[:partial_train_index])
validation_scores = np.array(scores[partial_train_index:]) # the tale of a misplaced colon costing me an hour

PERCENTILE = len(train_list)//100
BOTTOM_TEN_PERCENT_POSITIONS = 10*PERCENTILE # positions where black is most winning
TOP_TEN_PERCENT_POSITIONS = 95*PERCENTILE
MIDDLE_BOTTOM_POSITIONS = 45*PERCENTILE
MIDDLE_TOP_POSITIONS = 55*PERCENTILE

training_positions = np.array(sorted_positions[:BOTTOM_TEN_PERCENT_POSITIONS] + sorted_positions[MIDDLE_BOTTOM_POSITIONS:MIDDLE_TOP_POSITIONS] + sorted_positions[TOP_TEN_PERCENT_POSITIONS:])
training_scores = np.array(sorted_scores[:BOTTOM_TEN_PERCENT_POSITIONS] + sorted_scores[MIDDLE_BOTTOM_POSITIONS:MIDDLE_TOP_POSITIONS] + sorted_scores[TOP_TEN_PERCENT_POSITIONS:])

print(training_positions.shape, "tp shape", type(training_positions)) # (100, 936)
print(training_scores.shape, "ts shape", type(training_scores)) # (100,) So this should work, but for some reason it's complaining x doesn't fit y.
print(training_positions[0].shape, "one example shape")


print(conv_model.summary())


history = conv_model.fit(training_positions,
                        training_scores,
                        epochs=2,
                        batch_size=64,
                        validation_data=(validation_positions,validation_scores),
                        shuffle=True)

conv_model.save('conv_model.h5') # saves the whole mdodel into this file


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
    #standardized_score = (float(score.white().score()) - scores_mean)/scores_std
    my_board = np.array(np_board.board)
    prediction = conv_model.predict(np.array([my_board,])) # Ok so it was expecting a list of predictions, for a single prediction use this
    print("score:",score,"prediction:",prediction)
    #numerical_score = get_numerical_score(score)

data = pd.DataFrame(training_scores)
print(data.describe())


print("done")