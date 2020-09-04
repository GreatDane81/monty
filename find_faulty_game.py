# finding the game that failed on 1135: which is really the 1137th game, because starts at 0, failed at 1136
import Board # had to change python.linting.pylintEnable to false in settings.
# weird, looks like a pyLint thing.
from Game import Game

import chess
from chess.engine import Cp, Mate, MateGiven
import chess.pgn

import pickle

train_path_tal = "C:/Users/Ethan Dain/Desktop/University/Machine Learning/Code/monty/Tal.pgn"
tal_file = open(train_path_tal)

start = 0
stop = 9
while start < stop:
    game = chess.pgn.read_game(tal_file)
    start += 1
game = chess.pgn.read_game(tal_file)
print(game.headers["Date"])
print(game.headers["White"])
print(game.headers["Black"])
np_board =  Board.Board()
py_board = chess.Board()
move_num = 0
for move in game.mainline_moves():
    # update the np board
    Game.play_move_on_np_board(np_board, move)
    print(move)
    print(np_board)
    # update the py board
    py_board.push(move)
    #score =  engine.analyse(py_board, chess.engine.Limit(time=ANALYSIS_TIME))["score"]
    #numerical_score = get_numerical_score(score)
    move_num += 1
