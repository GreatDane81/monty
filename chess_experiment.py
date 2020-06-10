import chess
import chess.pgn

import numpy as np

import tensorflow as tf



ROWS = 9 # 8 for rows, + 1 for storing state. Ugly, but necessary for keras.
COLUMNS = 8 # one per col
PIECES = 13 # 6 for white, 6 for black + 1 for empty/full bit. 

# Storing these in row 9, will be important to engine eval of a lot of positions.
# Notice that this effectively only stores whether or not the king has moved,
# otherwise legal_moves() takes care of the rest
WHITE_CASTLE_BIT = 1 # will be at board[8, 1, 0]
BLACK_CASTLE_BIT = 1 # [8, 2, 0]
# pro tip, can use a PLAYER_TURN*6 + PIECE_SHIFT to address the pieces vector.

# n for knight for disambuity, I'm sorry
PIECE_OFFSET = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6} # offset of 1 for pawn because of empty bit
PIECE_TO_CHAR = {1:'p', 2:'n', 3:'b', 4:'r', 5:'q', 6:'k',7:'P', 8:'N', 9:'B', 10:'R', 11:'Q', 12:'K'}

def initialize_np_board():
    '''
    Initializes the chess board in Numpy representation
    '''
    # Initialize shapes with zeros
    board = np.zeros([ROWS, COLUMNS, PIECES],dtype='uint8')
    board[0:2, :, 0] = 1 # filling the first two rows with "full" bits (represents white)
    board[6:8, :, 0] = 1 # filling the last two rows with "full bits" (represents black)
    # Turn starts off as white, so no need to set the turn bit (board[8:0:0]) to 1.
    # set the white pawn row
    board[1, :, PIECE_OFFSET['p']] = 1
    # set the black pawn row
    board[6, :, 6 + PIECE_OFFSET['p']] = 1 # + 6 because they are black pawns
    # set the back rows for both white and black
    set_back_rank(board, 'w')
    set_back_rank(board, 'b')
    # Making sure both sides can castle:
    board[8, 1, 0] = WHITE_CASTLE_BIT
    board[8, 2, 0] = BLACK_CASTLE_BIT
    # Working as intended
    #print(board)
    return board

def set_back_rank(board, colour):
    if colour == 'w':
        colour_offset = 0
        row = 0
    else:
        colour_offset = 1
        row = 7
    # Take the first piece in the row, the rook, and find its index.
    # if the colour is black, shift 6 down and set the correct bit to 1
    board[row, 0, PIECE_OFFSET['r'] + 6*colour_offset] = 1
    board[row, 1, PIECE_OFFSET['n'] + 6*colour_offset] = 1
    board[row, 2, PIECE_OFFSET['b'] + 6*colour_offset] = 1
    board[row, 3, PIECE_OFFSET['q'] + 6*colour_offset] = 1
    board[row, 4, PIECE_OFFSET['k'] + 6*colour_offset] = 1
    board[row, 5, PIECE_OFFSET['b'] + 6*colour_offset] = 1
    board[row, 6, PIECE_OFFSET['n'] + 6*colour_offset] = 1
    board[row, 7, PIECE_OFFSET['r'] + 6*colour_offset] = 1

def board_to_str(board):
    '''
    Converts the numpy board into a human readable string
    '''
    #white_k_has_moved = board[8, 1, 0] & True # (anding for readability)
    #black_k_has_moved = board[8, 2, 0] & True
    #turn = board[8, 0, 0] & True
    s = ""
    for i in range(7, -1, -1): # kind of weird, but range is not inclusive
        for j in range(7, 0, -1):
            if board[i, j, 0] == 0: # Then square is empty, add a '.'
                s += '.'
            else:
                # get the piece
                for k in range (1,13): 
                    if board[i, j, k] == 1:
                        # found the piece
                        s += PIECE_TO_CHAR[k]
        s += "\n" # newline
    return s


file_path = 'C:/Users/Ethan Dain/Desktop/University/Machine Learning/Code/monty/kasparov-deep-blue-1997.pgn'
file = open(file_path)

board = initialize_np_board()

print(board_to_str(board))


first_game = chess.pgn.read_game(file)
print(first_game.headers["Event"])


board = first_game.board()
for move in first_game.mainline_moves():
    #print(move) # perfect, i don't even have to parse. Well, kind of.
    # have to only interpret squares to the tensor.
    pass

file.close()
print('done')

