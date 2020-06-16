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
LETTER_TO_ROW = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4,'f':5, 'g':6,'h':7}
PIECE_TO_BARRAY = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6,'P':7, 'N':8, 'B':9, 'R':10, 'Q':11, 'K':12}

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
        # Do king and queen separately
        board[row, 3, PIECE_OFFSET['q'] + 6*colour_offset] = 1
        board[row, 4, PIECE_OFFSET['k'] + 6*colour_offset] = 1
    else:
        colour_offset = 1
        row = 7
        board[row, 3, PIECE_OFFSET['q'] + 6*colour_offset] = 1
        board[row, 4, PIECE_OFFSET['k'] + 6*colour_offset] = 1
    # Take the first piece in the row, the rook, and find its index.
    # if the colour is black, shift 6 down and set the correct bit to 1
    board[row, 0, PIECE_OFFSET['r'] + 6*colour_offset] = 1
    board[row, 1, PIECE_OFFSET['n'] + 6*colour_offset] = 1
    board[row, 2, PIECE_OFFSET['b'] + 6*colour_offset] = 1
    # did the middle rows above
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
    for row in range(0, 8):
        s = row_to_str(row) + "\n" + s
    return s

def row_to_str(row_num):
    s = ""
    for col in range(0, 8):
        if board[row_num, col, 0] == 0:
            s += "."
        else:
            s += get_piece_from_index(board, row_num, col)
    return s


def get_board_tensor(board):
    '''
    Numpy Board --> Constant tensor of the board
    '''
    return tf.constant(board, dtype=tf.uint8)

def play_move(board, move, promotion):
    '''
    (Numpy, str) -> Numpy
    Takes a Numpy board, updates it given a move from move in game.mainline_moves()
    '''
    # TODO: Take care of special cases like promotion, castling, and maybe en passant.
    # first, parse the move string
    # TODO: This includes updating the bit map for castling.
    indices = parse_move_indices(move)
    start_index = indices[0]
    target_index = indices[1]
    # get the piece from the start index
    piece = get_piece_from_index(board, start_index[0], start_index[1])
    # next, empty the index from the start index
    empty_index(board, start_index[0], start_index[1])
    # then, update the target index
    if promotion != None:
        # Then get the promotion piece
        piece = PIECE_TO_CHAR[promotion]
    update_index(board, target_index[0], target_index[1], piece)
    # And switch player turns
    update_turn(board)

def parse_move_indices(move):
    '''
    returns the index for start piece, target piece
    --> [(s_row, s_col), (t_row, t_col)]
    NOTE: only takes the first chars for the indices, promotions and mate not considered
    '''
    move_str = str(move)
    start_sq = move_str[0:2]
    target_sq = move_str[2:4]
    # g1 should be 0, 6
    start_index = (int(start_sq[1]) - 1, LETTER_TO_ROW[start_sq[0]])
    target_index = (int(target_sq[1]) -1, LETTER_TO_ROW[target_sq[0]])
    return [start_index, target_index]
        

def get_piece_from_index(board, row, col):
    '''
    (row, col) -> str

    A1 -> 0, 0
    TODO: Clean up this code, board should be an argument
    '''
    barray = board[row, col]
    return get_piece_from_bit_array(board, barray)

def get_piece_from_bit_array(board, barray):
    '''
    bit array for pieces --> piece rep in str
    '''
    if barray[0] == 0:
        return None # TODO: error
    for k in range (1,13): 
        if barray[k] == 1:
            # found the piece
            return PIECE_TO_CHAR[k]

def empty_index(board, row, col):
    '''
    Empties the square at [row, col].
    Intended for pieces only, don't apply to row 9.
    '''
    # set the bit array to 0 (selection bit will be empty, as will all pieces)
    board[row,col] = np.zeros(PIECES)
    return

def update_index(board, row, col, piece):
    '''
    Writes over the square at row, col, with 'piece'.
    Note: Old piece is wiped, this accounts for captures.
    '''
    # Empty the board's pieces before updating
    board[row, col] = np.zeros(PIECES)
    # first, set the bit map to contain a piece
    board[row, col, 0] = 1
    # next, find the shift using PIECE_TO_SHIFT
    board[row, col, PIECE_TO_BARRAY[piece]] = 1

def castle_king_side(board, colour):
    '''
    Castles kingside given a colour. Returns None on failure
    TODO: Proper exception, optimize if else branching
    Again a big assumption here is that this move is LEGAL to play
    '''
    if colour == 'w':
        # First, check that the king and  kingside rook are in the right place
        king_index = parse_index('e1')
        rook_index = parse_index('h1')
        king_target = parse_index('g1')
        rook_target = parse_index('f1')
        king_char = 'k'
        rook_char = 'r'
    else:
        king_index = parse_index('e8')
        rook_index = parse_index('h8')
        king_target = parse_index('g8')
        rook_target = parse_index('f8')
        king_char = 'K'
        rook_char = 'R'
    castling_logic(board, king_index, rook_index, king_target, rook_target, king_char, rook_char)
    

def castle_queen_side(board, colour):
    '''
    Castles queenside according to colour
    '''
    if colour == 'w':
        king_index = parse_index('e1')
        rook_index = parse_index('a1')
        king_target = parse_index('c1')
        rook_target = parse_index('d1')
        king_char = 'k'
        rook_char = 'r'
    else:
        king_index = parse_index('e8')
        rook_index = parse_index('a8')
        king_target = parse_index('c8')
        rook_target = parse_index('d8')
        king_char = 'K'
        rook_char = 'R'
    castling_logic(board, king_index, rook_index, king_target, rook_target, king_char, rook_char)

def castling_logic(board, king_index, rook_index, king_target, rook_target, king_char, rook_char):
    '''
    Castling logic that will be called on all castles: white/black kingside, white/black queenside.
    '''
    king_correct = get_piece_from_index(board, king_index[0], king_index[1]) == king_char
    if not king_correct:
       return None # TODO: fail out
    rook_correct = get_piece_from_index(board, rook_index[0], rook_index[1]) == rook_char
    if not rook_correct:
        return None
    # Passed error handling, update the pieces
    empty_index(board, king_index[0], king_index[1])
    empty_index(board, rook_index[0], rook_index[1])
    # update the pieces
    update_index(board, king_target[0], king_target[1], king_char)
    update_index(board, rook_target[0], rook_target[1], rook_char)
    update_turn(board)
    # update castling bit
    castle_update(board, king_char)


def parse_index(index):
    '''
    "a1" --> (0, 0)
    TODO: update to use in other functions that need it
    '''
    # TODO: Assert length = 2
    return (int(index[1]) - 1, LETTER_TO_ROW[index[0]])

def update_turn(board):
    board[8,0,0] = 1 - board[8,0,0]

def white_can_castle(board):
    return board[8, 1, 0] == 1

def black_can_castle(board):
    return board[8,2,0] == 1

def castle_update(board, king):
    if king == 'k':
        board[8,1, 0] = 0
    else:
        board[8,2,0] = 0


#file_path = 'C:/Users/Ethan Dain/Desktop/University/Machine Learning/Code/monty/kasparov-deep-blue-1997.pgn'
file_path = 'C:/Users/Ethan Dain/Desktop/University/Machine Learning/Code/monty/promotion.pgn'
file = open(file_path)

board = initialize_np_board()

board_tensor = get_board_tensor(board)

#print(board_to_str(board))


first_game = chess.pgn.read_game(file)
print(first_game.headers["Event"])

print(board_to_str(board))
file_obj = open("C:/Users/Ethan Dain/Desktop/University/Machine Learning/Code/monty/board_debug.txt","w")

for move in first_game.mainline_moves():
    try:
        # detection of white kingside castle
        move_str = str(move)
        if white_can_castle(board) and move_str == 'e1g1':
            # Then king is castling. Error handling:
            castle_king_side(board, 'w')
            file_obj.writelines(str(move) +" white k side castle \n")
        elif black_can_castle(board) and move_str == 'e8g8':
            castle_king_side(board, 'b')
            file_obj.writelines(str(move) +" black k side castle \n")
        else:
            play_move(board, move, move.promotion)
            file_obj.writelines(str(move) +"\n")
        file_obj.writelines(board_to_str(board))
        file_obj.writelines("\n\n")
    except:
        print("failed out")
        break
file_obj.close()


#play_move(board, 'd7d5') # also works after g1f3
#print(board_to_str(board)) 

#print(get_piece_from_index(2,5)) n g1 to f3 working as expected
#print(get_piece_from_index(0,3))

file.close()
print('done')

