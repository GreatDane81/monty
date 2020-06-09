import chess
import chess.pgn

import tensorflow as tf



ROWS = 9 # 8 for rows, + 1 for storing state. Ugly, but necessary for keras.
COLUMNS = 8 # one per col
PIECES = 12 # 6 for white, 6 for black. 
# pro tip, can use a PLAYER_TURN*6 + PIECE_SHIFT to address the pieces vector.


def initialize_tensor_board():
    # start with 0s
    # Hm need to research what is computationally faster.
    # Two options here. Either I have one tensor which I update perpetually throughout a single game
    # or I keep a python nested list to represent the matrix and write over it every time a move is changed.
    # This could make a big difference computationally. 
    # I'm leaning towards making it constant on each input, because dealing with python lists seems easier.
    # i'm also sure there are computational advantages to using the constant class. 
    board = tf.Variable(tf.zeros([ROWS, COLUMNS, PIECES], dtype=tf.uint8),dtype=tf.uint8)

    print(board[0, 0])

file_path = 'C:/Users/Ethan Dain/Desktop/University/Machine Learning/Code/monty/kasparov-deep-blue-1997.pgn'
file = open(file_path)

initialize_tensor_board()

first_game = chess.pgn.read_game(file)
print(first_game.headers["Event"])


board = first_game.board()
for move in first_game.mainline_moves():
    #print(move) # perfect, i don't even have to parse. Well, kind of.
    # have to only interpret squares to the tensor.
    pass

file.close()
print('done')

