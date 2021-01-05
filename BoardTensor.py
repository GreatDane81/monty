# This file is intentionally built as a set of functions.
# It doesn't make sense to have a class-based representation if I intend each board
# to be an immutable, 'snapshot', representation of a python chess board.

NUM_ROWS = 8
NUM_COLS = 8
NUM_DIMS = 7 # Really this is 6 + 1 (6 for each type of piece, + 1 for heurestics layer)



def board_to_tensor(board):
    """
    (chess.Board) -> (8*8*7 Tensor)

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
    # This function will be implemented by looping through each square of the given board and filling in
    # the requisite information

