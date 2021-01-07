import chess
import numpy as np

# This file is intentionally built as a set of functions.
# It doesn't make sense to have a class-based representation if I intend each board
# to be an immutable, 'snapshot', representation of a python chess board.

NUM_ROWS = 8
NUM_COLS = 8
NUM_DIMS = 7 # Really this is 6 + 1 (6 for each type of piece, + 1 for heurestics layer)


# the values that will represent the presence of a black/white square
WHITE_PRESENT = 1
BLACK_PRESENT = -1



# Turn bit location in the heuristics layer
TURN_BIT_INDEX = (0,0)


# maps the piece type to its layer
PIECE_TYPE_TO_LAYER = {chess.KING:0, chess.PAWN:1, chess.ROOK:2, chess.QUEEN:3, chess.BISHOP:4, chess.KNIGHT: 5}

# maps the layer to the outchar
LAYER_TO_CHAR = {0:'k', 1:'p', 2:'r', 3:'q', 4:'b', 5:'n'}

# heurestics dimension:
HEURISTICS_LAYER = 6


def board_to_tensor(board):
    """
    (chess.Board) -> 8*8*7 NP Tensor

    Takes a python-chess Board, 'board' as input and returns an 8*8*7 Tensor representation of that Board.

    The output Tensor's structure as follows:
    - (0,0) Represents A1, and (7,7) represents H8.
    - Each of the 6 following dimensions represent whether or not a piece of a certain type is located on
      that square. For example: If the Tensor's value at [0,0,0] is 1, this means the white King is at A1,
      because the king layer's bit is 1. This value will be -1 if the black king resides at A1.
    Layers:
        - King: 0
        - Pawn: 1
        - Rook: 2
        - Queen: 3
        - Bishop: 4
        - Knight: 5
        - "heurestics": 6
    "heurestics" layer is independent of the rest of the layers.
        - at [0,0,6] the Turn bit resides. 1 for white, -1 for black. 
    """
    # use the smallest datatype supported by keras
    tensor = np.zeros([NUM_ROWS, NUM_COLS, NUM_DIMS], dtype='float16')
    # This function will be implemented by looping through each square of the given board and filling in
    # the requisite information
    for square in chess.SQUARES:
        # First, check if a piece is at that square
        piece = board.piece_at(square)
        if piece is not None:
            # Then there is a colour, and we need to perform our analysis for pieces
            piece_layer = PIECE_TYPE_TO_LAYER[piece.piece_type]
            # And we will need to update the tensor in the corresponding place
            col, row = chess.square_file(square), chess.square_rank(square)
            # set the right place colour
            if piece.color == chess.WHITE:
                place_colour = WHITE_PRESENT
            else:
                place_colour = BLACK_PRESENT
            # now we can fill in the tensor
            tensor[row, col, piece_layer] = place_colour
    # and finish by setting the heuristic values
    if board.turn == chess.WHITE:
        turn = WHITE_PRESENT
    else:
        turn = BLACK_PRESENT
    tensor[TURN_BIT_INDEX[0], TURN_BIT_INDEX[1], HEURISTICS_LAYER] = turn
    return tensor

def print_tensor(tensor):
    """
    (8*8*7 NP Tensor) -> None

    Prints the given NP tensor, for verification purposes.

    Lowercase represents white, uppercase represents black
    """
    s = ""
    for row in range(7, -1, -1):
        for col in range(0, 8):
            # print top down, left to right
            out_char = "." # assume no piece
            for piece in range(0, 6):
                if tensor[row, col, piece] != 0:
                    # a piece of this type was found
                    out_char = LAYER_TO_CHAR[piece]
                    if tensor[row,col, piece] == BLACK_PRESENT:
                        out_char = out_char.upper() # capitalize it
            # after each square has been analysed
            s += out_char
        # after each row print a new line
        s += "\n"
    # after all, print s
    print(s)
                     



if __name__ == "__main__":
    # Some testing stuff to verify the conversion is working as intended
    board = chess.Board()
    board.push(chess.Move.from_uci("d2d4"))
    x = board_to_tensor(board)
    print_tensor(x)
    board.push(chess.Move.from_uci("e7e5"))
    x = board_to_tensor(board)
    print_tensor(x)
    board.push(chess.Move.from_uci("d4e5"))

    x = board_to_tensor(board)
    print_tensor(x)


        




