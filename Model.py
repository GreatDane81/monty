import generate_train_data

from tensorflow import keras 


import numpy as np

# for conv_model:
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

from Game import Game
from GameTwoD import GameTwoD
import Board
import TwoDBoard

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


np_board_shape = (9, 8) # updated


conv_model = Sequential()
inp = keras.layers.Input(batch_input_shape=(None,9,8))# ok just assume 13 channels for now
conv_model.add(inp)
conv_model.add(Dense(128, activation='relu')) # TODO: the one here is usually for grayscale images, not sure how this will work with 1/0 bin
#conv_model.add(Dropout(rate=0.2))
conv_model.add(Flatten())
conv_model.add(Dense(1))

optimizer = keras.optimizers.Adam()

conv_model.compile(optimizer=optimizer,loss='mean_absolute_error')  # maybe the idea should be use functions that identify "average case" which are by far
# more common that outlier "totally winning" situations to fit?
print("compiled successfully")



# getting the training data
out_file_path = "C:/Users/Ethan/Documents/GitHub/monty/training_out_file"
added_data_out_file_path = "C:/Users/Ethan/Documents/GitHub/monty/added_data_out_file"
train_list = generate_train_data.load_training_data(added_data_out_file_path)

# going to try sorting by scores, and select data by each quintile in equal proportions

sorted_train_list = sorted(train_list,key=itemgetter(1))

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

positions = np.array(positions)
scores = np.array(scores)


scores_sum = np.sum(scores)
scores_mean = np.mean(scores)
scores_std = np.std(scores)

#standardize data to make training the model easier 
#for i in range(0, len(scores)):
#    scores[i] = (scores[i] - scores_mean)/scores_std


PERCENTILE = len(train_list)//100
BOTTOM_PERCENT_POSITIONS = 1*PERCENTILE # positions where black is most winning
TOP_PERCENT_POSITIONS = 1*PERCENTILE
MIDDLE_BOTTOM_POSITIONS = 0*PERCENTILE
MIDDLE_TOP_POSITIONS = 0*PERCENTILE

sorted_positions = sorted_positions[:BOTTOM_PERCENT_POSITIONS] + sorted_positions[MIDDLE_BOTTOM_POSITIONS:MIDDLE_TOP_POSITIONS] + sorted_positions[TOP_PERCENT_POSITIONS:]
sorted_scores = sorted_scores[:BOTTOM_PERCENT_POSITIONS] + sorted_scores[MIDDLE_BOTTOM_POSITIONS:MIDDLE_TOP_POSITIONS] + sorted_scores[TOP_PERCENT_POSITIONS:]

sorted_positions= np.array(sorted_positions)
sorted_scores = np.array(sorted_scores)

# normalize sorted scores:
#standardize data to make training the model easier 
#for i in range(0, len(sorted_scores)):
#    sorted_scores[i] = (sorted_scores[i] - scores_mean)/scores_std



print(conv_model.summary())

history = conv_model.fit(sorted_positions,
                        sorted_scores,
                        epochs=2,
                        batch_size=1,
                        validation_split=0.2,
                        shuffle=True)

conv_model.save('conv_model.h5') # saves the whole mdodel into this file


train_path_tal = "C:/Users/Ethan/Documents/GitHub/monty/Tal.pgn"
tal_file = open(train_path_tal)

game = chess.pgn.read_game(tal_file)

np_board =  TwoDBoard.TwoDBoard()
py_board = chess.Board()


SF_DEPTH = 10 # depth for stockfish

ANALYSIS_TIME = 0.1 # in seconds

path = "C:/Users/Ethan/Documents/GitHub/monty/stockfish/stockfish-11-win/stockfish-11-win/Windows/stockfish_20011801_x64"

engine = chess.engine.SimpleEngine.popen_uci(path)

for move in game.mainline_moves():
    # update the np board
    GameTwoD.play_move_on_np_board(np_board, move)
    py_board.push(move)
    score =  engine.analyse(py_board, chess.engine.Limit(time=ANALYSIS_TIME))["score"]
    my_board = np.array(np_board.board)
    prediction = conv_model.predict(np.array([my_board,])) # Ok so it was expecting a list of predictions, for a single prediction use this
    print("score:",score,"prediction:",prediction)
    #numerical_score = get_numerical_score(score)

data = pd.DataFrame(scores)
print(data.describe())


print("done")