import numpy as np
# moved it all to class, have to test. Maybe clean up too.

ROWS = 8
COLUMNS = 8
LAYERS = 6 + 2 + 1 # 9: 6 for piece structure (one layer for each peice, 1/-1 for black/white), 2 for SQUARES_ATTACKED and HEURESTICS

# PIECE LAYERS:
KING_LAYER = 0
PAWN_LAYER = 1
QUEEN_LAYER = 2
ROOK_LAYER = 3
BISHOP_LAYER = 4
KNIGHT_LAYER = 5 # which is the last layer
LAST_LAYER = KNIGHT_LAYER

#TURNS
WHITES_TURN = 0
BLACKS_TURN = 1

# ATTACK LAYERS:
WHITE_ATTACK_LAYER = 6
BLACK_ATTACK_LAYER = 7

# HEURISTICS LAYER
HEURESTICS_LAYER = 8
# HEURISTICS BIT SHIFTS
TURN_INDEX = [0,0] # going to leave the rest of the row blank for pattern recognition
WHITE_KSIDE_CASTLE_INDEX = [1,0] # has to be two dimensions becuase hereustics is now a layer
WHITE_QSIDE_CASTLE_INDEX = [1,1]
BLACK_KSIDE_CASTLE_INDEX = [1,2]
BLACK_QSIDE_CASTLE_INDEX = [1,3]

# PRESENT BITS
WHITE_PRESENT = 1
BLACK_PRESENT = -1

# LAYER TO STR
WHITE_LAYER_TO_PIECE = {KING_LAYER:'k', QUEEN_LAYER: 'q', ROOK_LAYER:'r', BISHOP_LAYER:'b', KNIGHT_LAYER: 'n', PAWN_LAYER: 'p'}
BLACK_LAYER_TO_PIECE = {KING_LAYER:'K', QUEEN_LAYER: 'Q', ROOK_LAYER:'R', BISHOP_LAYER:'B', KNIGHT_LAYER: 'N', PAWN_LAYER: 'P'}

# PIECE TO LAYER (all in one dict for useability in update_index)
PIECE_TO_LAYER = {'k': KING_LAYER, 'q': QUEEN_LAYER, 'r': ROOK_LAYER, 'b': BISHOP_LAYER, 'n': KNIGHT_LAYER, 'p': PAWN_LAYER,
                  'K': KING_LAYER, 'Q': QUEEN_LAYER, 'R': ROOK_LAYER, 'B': BISHOP_LAYER, 'N': KNIGHT_LAYER, 'P': PAWN_LAYER}
                                                                                                                            
# For promotion and conversion of types:
PYCHESS_PIECE_TYPE_TO_BOARD_PIECE_TYPE = {1:'p', 2:'n', 3:'b', 4:'r', 5:'q', 6:'k'}

# n for knight for disambuity, I'm sorry
LETTER_TO_ROW = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4,'f':5, 'g':6,'h':7}


class Board: 
    def __init__(self):
        '''
        Creates a new board with standard starting position
        '''
        # start with an empty board
        self.board = np.zeros([ROWS, COLUMNS, LAYERS],dtype='float32')
        # set up piece layers
        pieces_to_index = {'k':{'e1'}, 'q':{'d1'}, 'r':{'a1','h1'}, 'n':{'b1', 'g1'}, 'b':{'f1','c1'}, 'p':{'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'},
                           'K':{'e8'}, 'Q':{'d8'}, 'R':{'a8','h8'}, 'N':{'b8', 'g8'}, 'B':{'f8','c8'}, 'P':{'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7'}}
        self.set_up_pieces(pieces_to_index)
        # set up attack layers
        #self.white_attack_setup()
        #self.black_attack_setup()
        # set up heurestics layer
        #self.heurestics_setup()

    def set_up_pieces(self, pieces_to_index):
        """
        dict of (char: {char}), 'k': 'e1'
    
        To set up all the pieces in their appropriate layer
        """
        for piece in pieces_to_index:
            # get the piece's list
            piece_indices = pieces_to_index[piece]
            for index in piece_indices: # for every index it appears in
                index = Board.parse_index(index)
                self.update_index(index[0], index[1], piece)


    def __str__(self):
        '''
        Converts the numpy board into a human readable string
        '''
        s = ""
        for row in range(7, -1, -1):
            s += self._print_row(row) + "\n"
        return s
    
    def _print_row(self, row):
        s = ""
        for col in range(0, 8):
            piece = Board.get_piece_from_index(self.board, row, col)
            if piece == None:
                s += "."
            else:
                s += piece
        return s


    def get_constant_tensor(self):
        '''
        Numpy Board --> Constant tensor of the board
        '''
        return tf.constant(self.board, dtype=tf.float32)

    def play_move(self, move, promotion):
        '''
        (str, pychess ChessPiece) -> None

        Updates the board given coordinates for a move, promotes if necessary.
        '''
        # start by getting indices
        indices = Board.parse_move_indices(move)
        start_index = indices[0]
        target_index = indices[1]
        # get the piece from the start index
        piece = Board.get_piece_from_index(self.board, start_index[0], start_index[1])
        # check if the king is being moved. If yes, remove appropriate castle rights
        if piece == "k":
            self.set_white_no_castle("A") # redundant sometimes, but i think safety is key here
        if piece == "K":
            self.set_black_no_castle("A")
        if piece == "r" and start_index == Board.parse_index("h1"):
            self.set_white_no_castle("K")
        if piece == "r" and start_index == Board.parse_index("a1"):
            self.set_white_no_castle("Q")
        if piece == "R" and start_index == Board.parse_index("h8"):
            self.set_black_no_castle("K")
        if piece == "R" and start_index == Board.parse_index("a8"):
            self.set_black_no_castle("Q")
        # next, empty the index from the start index
        self.empty_index(start_index[0], start_index[1])
        # then, update the target index
        if promotion != None:
            # Then get the promotion piece
            py_chess_piece_type = promotion
            piece = PYCHESS_PIECE_TYPE_TO_BOARD_PIECE_TYPE[py_chess_piece_type]
            if self.get_turn() == BLACKS_TURN:
                # then take it to uppercase, because it's black's turn
                piece = piece.upper()
        self.update_index(target_index[0], target_index[1], piece)
        # And switch player turns
        self.update_turn()


    def empty_index(self, row, col):
        '''
        Empties the square at [row, col].

        Raises Exception if no piece at that index.
        '''
        # first, check there is a piece (error should propogate, so no try catch)
        piece_layer = Board.get_piece_layer_from_index(self.board, row, col)
        # get the piece
        piece = self.board[piece_layer, row, col]
        print(piece.shape)
        #if piece == 0:
        #    raise Exception("Tried to empty a non-existent piece at row,", row, "col,", col)

    def update_index(self, row, col, piece):
        '''
        (int, int, char)
        Writes over the square at row, col, with 'piece'.
        Note: Old piece is wiped, this accounts for captures.
        '''
        # First, empty the target index to ensure no two layers save an overlapping index
        # (I.E) both knight and bishop occupy a square
        self.empty_index(row, col)
        # next, find the appropriate layer to put in the new piece
        layer = PIECE_TO_LAYER[piece]
        if piece.islower():
            # then update with WHITE_PRESENT
            self.board[layer, row, col] = WHITE_PRESENT
        else:
            self.board[layer, row, col] = BLACK_PRESENT


    def castle_king_side(self, colour):
        '''
        Castles kingside given a colour. Returns None on failure
        TODO: Proper exception, optimize if else branching
        Again a big assumption here is that this move is LEGAL to play
        '''
        if colour == 'w':
            # First, check that the king and  kingside rook are in the right place
            king_index = Board.parse_index('e1')
            rook_index = Board.parse_index('h1')
            king_target = Board.parse_index('g1')
            rook_target = Board.parse_index('f1')
            king_char = 'k'
            rook_char = 'r'
        else:
            king_index = Board.parse_index('e8')
            rook_index = Board.parse_index('h8')
            king_target = Board.parse_index('g8')
            rook_target = Board.parse_index('f8')
            king_char = 'K'
            rook_char = 'R'
        self.castling_logic(king_index, rook_index, king_target, rook_target, king_char, rook_char)
        

    def castle_queen_side(self, colour):
        '''
        Castles queenside according to colour
        '''
        if colour == 'w':
            king_index = Board.parse_index('e1')
            rook_index = Board.parse_index('a1')
            king_target = Board.parse_index('c1')
            rook_target = Board.parse_index('d1')
            king_char = 'k'
            rook_char = 'r'
        else:
            king_index = Board.parse_index('e8')
            rook_index = Board.parse_index('a8')
            king_target = Board.parse_index('c8')
            rook_target = Board.parse_index('d8')
            king_char = 'K'
            rook_char = 'R'
        self.castling_logic(king_index, rook_index, king_target, rook_target, king_char, rook_char)

    def castling_logic(self, king_index, rook_index, king_target, rook_target, king_char, rook_char):
        '''
        Castling logic that will be called on all castles: white/black kingside, white/black queenside.
        '''
        king_correct = Board.get_piece_from_index(self.board, king_index[0], king_index[1]) == king_char
        if not king_correct:
            return None # TODO: fail out
        rook_correct = Board.get_piece_from_index(self.board, rook_index[0], rook_index[1]) == rook_char
        if not rook_correct:
            return None
        # Passed error handling, update the pieces
        self.empty_index(king_index[0], king_index[1])
        self.empty_index(rook_index[0], rook_index[1])
        # update the pieces
        self.update_index(king_target[0], king_target[1], king_char)
        self.update_index(rook_target[0], rook_target[1], rook_char)
        # update castling bit
        if king_char == "k":
            self.set_white_no_castle("A")
        else:
            self.set_black_no_castle("A")
        # conclude by updating the turn
        self.update_turn()

    def update_turn(self):
        """
        Flips the turn bit
        """
        turn = self.board[HEURESTICS_LAYER,TURN_INDEX[0],TURN_INDEX[1]]
        self.board[HEURESTICS_LAYER,TURN_INDEX[0],TURN_INDEX[1]] = 1 - turn

    def white_can_castle(self, side):
        """
        (char) -> None

        A for either side
        K for kingside, Q for Q side
        """
        kside = self.board[HEURISTICS_LAYER, WHITE_KSIDE_CASTLE_INDEX[0], WHITE_KSIDE_CASTLE_INDEX[1]] == 1
        qside = self.board[HEURISTICS_LAYER, WHITE_QSIDE_CASTLE_INDEX[0], WHITE_QSIDE_CASTLE_INDEX[1]] == 1
        if side == "A":
            return kside or qside
        if side == "K":
            return kside
        else:
            return qside 
    
    def set_white_no_castle(self, type):
        """
        (char) -> None

        A: for no castling at all
        K: for no kingside
        Q: for no queenside
        """
        if type == "A":
            self.board[HEURISTICS_LAYER, WHITE_KSIDE_CASTLE_INDEX[0], WHITE_KSIDE_CASTLE_INDEX[1]] = 0
            self.board[HEURISTICS_LAYER, WHITE_KSIDE_CASTLE_INDEX[0], WHITE_KSIDE_CASTLE_INDEX[1]] = 0
        elif type == "K":
                self.board[HEURESTICS_LAYER, WHITE_KSIDE_CASTLE_INDEX[0], WHITE_KSIDE_CASTLE_INDEX[1]] = 0 # set the white kside to 0
        else:
            self.board[HEURESTICS_LAYER,WHITE_KSIDE_CASTLE_INDEX[0],WHITE_KSIDE_CASTLE_INDEX[1]] = 0

    def black_can_castle(self, side):
        """
        (char) -> None

        A for either side
        K for kingside, Q for Q side
        """
        kside = self.board[HEURISTICS_LAYER, BLACK_KSIDE_CASTLE_INDEX[0], BLACK_KSIDE_CASTLE_INDEX[1]] == 1
        qside = self.board[HEURISTICS_LAYER, BLACK_QSIDE_CASTLE_INDEX[0], BLACK_QSIDE_CASTLE_INDEX[1]] == 1
        if side == "A":
            return kside or qside
        if side == "K":
            return kside
        else:
            return qside 
    
    def set_black_no_castle(self, side):
        """
        A: for no castling at all
        K: for no kingside
        Q: for no queenside
        """
        if type == "A":
            self.board[HEURISTICS_LAYER, BLACK_KSIDE_CASTLE_INDEX[0], BLACK_KSIDE_CASTLE_INDEX[1]] = 0
            self.board[HEURISTICS_LAYER, BLACK_KSIDE_CASTLE_INDEX[0], BLACK_KSIDE_CASTLE_INDEX[1]] = 0
        elif type == "K":
                self.board[HEURESTICS_LAYER, BLACK_KSIDE_CASTLE_INDEX[0], BLACK_KSIDE_CASTLE_INDEX[1]] = 0 # set the white kside to 0
        else:
            self.board[HEURESTICS_LAYER,BLACK_KSIDE_CASTLE_INDEX[0],BLACK_KSIDE_CASTLE_INDEX[1]] = 0
    
    @staticmethod
    def parse_index(index):
        '''
        "a1" --> (0, 0)
        '''
        # TODO: Assert length = 2
        return (int(index[1]) - 1, LETTER_TO_ROW[index[0]])

    @staticmethod
    def get_piece_layer_from_index(board, row, col):
        """
        (int, int) -> layer the piece resides in, independent of colour.

        Returns None if no piece found
        """
        # Here we're going to have to traverse through the 6 piece layers
        for layer in range(0, LAST_LAYER + 1): # because the knight is the last layer
            if board[layer, row, col] != 0:
                # Then the piece resides here
                return layer
        return None

    @staticmethod
    def get_piece_from_index(board, row, col):
        '''
        (row, col) -> str
        '''
        # (Exception propogates)
        # get the layer
        piece_layer = Board.get_piece_layer_from_index(board, row, col)
        if piece_layer == None:
            return None
        piece = board[piece_layer, row, col]
        if piece == WHITE_PRESENT:
            return WHITE_LAYER_TO_PIECE[piece_layer]
        else:
            return BLACK_LAYER_TO_PIECE[piece_layer]

    
    @staticmethod
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
        start_index = Board.parse_index(start_sq)
        target_index = Board.parse_index(target_sq)
        return [start_index, target_index]


board = Board()

print(board)