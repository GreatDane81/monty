# for pychess
import chess
from chess.engine import Cp, Mate, MateGiven
import chess.pgn

# for keras
from tensorflow import keras

# for numpy
import numpy as np

# for copy
import copy

# for my stuff
import Board

def select_move(py_board, game_board):
    '''
    Looks at all legal moves, picks best one

    Currently for black
    '''
    legal_moves = game_board.legal_moves
    best_score = 1000000 # just some really high number for white leading
    best_move = None
    save_board = copy.deepcopy(py_board)
    for move in legal_moves:
        move_str = str(move)
        py_board.play_move(move_str, move.promotion)
        my_board = py_board.board
        # get the score
        current_score = model.predict(np.array([my_board,]))
        if current_score < best_score:
            best_score = current_score
            best_move = move
        # and reset the py_board
        py_board = copy.deepcopy(save_board)
    return (best_move, best_score, py_board)


game_board = chess.Board()
python_board = Board.Board()

model = keras.models.load_model("conv_model.h5")



#print("before result")
#print(python_board)
#result = select_move(python_board, game_board)
#monty_move = result[0]
python_board = result[2]
#print("when taken from results")
#print(python_board)

#python_board_2 = Board.Board()
#python_board_2.play_move('e2e4', None)
#python_board_2.play_move('g8h6', None) # somehow this move is pushed but
# i can't for the life of me figure out why
# because the board before being returned looks good, then after being returned
# it's messed up.
#s=str(python_board)
#s1= str(python_board_2)
#print(s == s1)
#move = result[0]
#move_str = str(move)

# TODO: OK apparently the board gets corrupted somehow, somewhere if you don't return it's copy?
# even though without returning its copy it looks totally fine? definitely something to look into.
# weirdest bug i've seen in my life.

print("Welcome! You open with the white pieces.")
while True:
    user_input = input("Enter your move now:")
    user_move = chess.Move.from_uci(user_input)
    # push the move on the python board and the game board
    game_board.push(user_move)
    python_board.play_move(str(user_move), user_move.promotion)
    print(python_board)
    # predict the best move
    result = select_move(python_board, game_board)
    # after getting the result, push the move and update the board
    python_board = result[2] # TODO: This is the super weird update
    monty_move = result[0]
    print("Monty replied with:", monty_move)
    game_board.push(monty_move)
    python_board.play_move(str(monty_move), monty_move.promotion)
    print(python_board)




