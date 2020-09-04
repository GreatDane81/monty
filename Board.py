import numpy as np
# moved it all to class, have to test. Maybe clean up too.

ROWS = 9 # 8 for rows, + 1 for storing state. Ugly, but necessary for keras.
COLUMNS = 8 # one per col
PIECES = 13 # 6 for white, 6 for black + 1 for empty/full bit. 

# Storing these in row 9, will be important to engine eval of a lot of positions.
# Notice that this effectively only stores whether or not the king has moved,
# otherwise legal_moves() takes care of the rest

# change: all stored at [8, x, 0], but this shift determines where exactly
WHITE_KSIDE_CASTLE_SHIFT = 1 # will be at board[8,1,0]
WHITE_QSIDE_CASTLE_SHIFT = 2 # will be at board[8,2,0]
BLACK_KSIDE_CASTLE_SHIFT = 3 # will be at board[8,3,0]
BLACK_QSIDE_CASTLE_SHIFT = 4 # will be at board[8,4,0]
# pro tip, can use a PLAYER_TURN*6 + PIECE_SHIFT to address the pieces vector.

# n for knight for disambuity, I'm sorry
PIECE_OFFSET = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6} # offset of 1 for pawn because of empty bit
PIECE_TO_CHAR = {1:'p', 2:'n', 3:'b', 4:'r', 5:'q', 6:'k',7:'P', 8:'N', 9:'B', 10:'R', 11:'Q', 12:'K'}
LETTER_TO_ROW = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4,'f':5, 'g':6,'h':7}
PIECE_TO_BARRAY = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6,'P':7, 'N':8, 'B':9, 'R':10, 'Q':11, 'K':12}

class Board: 
    

    def __init__(self):
        '''
        Creates a new board with standard starting position
        '''
        self.board = np.zeros([ROWS, COLUMNS, PIECES],dtype='float32')
        self.board[0:2, :, 0] = 1 # filling the first two rows with "full" bits (represents white)
        self.board[6:8, :, 0] = 1 # filling the last two rows with "full bits" (represents black)
        # Turn starts off as white, so no need to set the turn bit (board[8:0:0]) to 1.
        # set the white pawn row
        self.board[1, :, PIECE_OFFSET['p']] = 1
        # set the black pawn row
        self.board[6, :, 6 + PIECE_OFFSET['p']] = 1 # + 6 because they are black pawns
        # set the back rows for both white and black
        self.set_back_rank('w')
        self.set_back_rank('b')
        # Making sure both sides can castle:
        self.board[8, WHITE_KSIDE_CASTLE_SHIFT, 0] = 1 # white_k_side
        self.board[8, WHITE_QSIDE_CASTLE_SHIFT, 0] = 1 # white_q_side
        self.board[8, BLACK_KSIDE_CASTLE_SHIFT, 0] = 1 # black k side
        self.board[8, BLACK_QSIDE_CASTLE_SHIFT, 0] = 1 # black q side

    def set_back_rank(self, colour):
        if colour == 'w':
            colour_offset = 0
            row = 0
            # Do king and queen separately
            self.board[row, 3, PIECE_OFFSET['q'] + 6*colour_offset] = 1
            self.board[row, 4, PIECE_OFFSET['k'] + 6*colour_offset] = 1
        else:
            colour_offset = 1
            row = 7
            self.board[row, 3, PIECE_OFFSET['q'] + 6*colour_offset] = 1
            self.board[row, 4, PIECE_OFFSET['k'] + 6*colour_offset] = 1
        # Take the first piece in the row, the rook, and find its index.
        # if the colour is black, shift 6 down and set the correct bit to 1
        self.board[row, 0, PIECE_OFFSET['r'] + 6*colour_offset] = 1
        self.board[row, 1, PIECE_OFFSET['n'] + 6*colour_offset] = 1
        self.board[row, 2, PIECE_OFFSET['b'] + 6*colour_offset] = 1
        # did the middle rows above
        self.board[row, 5, PIECE_OFFSET['b'] + 6*colour_offset] = 1
        self.board[row, 6, PIECE_OFFSET['n'] + 6*colour_offset] = 1
        self.board[row, 7, PIECE_OFFSET['r'] + 6*colour_offset] = 1

    def __str__(self):
        '''
        Converts the numpy board into a human readable string
        '''
        #white_k_has_moved = board[8, 1, 0] & True # (anding for readability)
        #black_k_has_moved = board[8, 2, 0] & True
        #turn = board[8, 0, 0] & True
        s = ""
        for row in range(0, 8):
            s = self.row_to_str(row) + "\n" + s
        return s

    def row_to_str(self, row_num):
        s = ""
        for col in range(0, 8):
            if self.board[row_num, col, 0] == 0:
                s += "."
            else:
                s += Board.get_piece_from_index(self.board, row_num, col)
        return s


    def get_board_tensor(self):
        '''
        Numpy Board --> Constant tensor of the board
        '''
        return tf.constant(self.board, dtype=tf.float32)

    def play_move(self, move, promotion):
        '''
        (Numpy, str) -> Numpy
        Takes a Numpy board, updates it given a move from move in game.mainline_moves()
        '''
        # TODO: Take care of special cases like promotion, castling, and maybe en passant.
        # first, parse the move string
        # TODO: This includes updating the bit map for castling.
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
            piece = PIECE_TO_CHAR[promotion]
        self.update_index(target_index[0], target_index[1], piece)
        # And switch player turns
        self.update_turn()
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
    
            
    @staticmethod
    def get_piece_from_index(board, row, col):
        '''
        (row, col) -> str

        A1 -> 0, 0
        TODO: Clean up this code, board should be an argument
        '''
        barray = board[row, col]
        return Board.get_piece_from_bit_array(board, barray)
    @staticmethod
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

    def empty_index(self, row, col):
        '''
        Empties the square at [row, col].
        Intended for pieces only, don't apply to row 9.
        '''
        # set the bit array to 0 (selection bit will be empty, as will all pieces)
        self.board[row,col] = np.zeros(PIECES)
        return

    def update_index(self, row, col, piece):
        '''
        Writes over the square at row, col, with 'piece'.
        Note: Old piece is wiped, this accounts for captures.
        '''
        # Empty the board's pieces before updating
        self.board[row, col] = np.zeros(PIECES)
        # first, set the bit map to contain a piece
        self.board[row, col, 0] = 1
        # next, find the shift using PIECE_TO_SHIFT
        self.board[row, col, PIECE_TO_BARRAY[piece]] = 1

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
        self.update_turn()
        # update castling bit
        if king_char == "k":
            self.set_white_no_castle("A")
        else:
            self.set_black_no_castle("A")

    @staticmethod
    def parse_index(index):
        '''
        "a1" --> (0, 0)
        TODO: update to use in other functions that need it
        '''
        # TODO: Assert length = 2
        return (int(index[1]) - 1, LETTER_TO_ROW[index[0]])


    def update_turn(self):
        self.board[8,0,0] = 1 - self.board[8,0,0]

    def white_can_castle(self, side):
        """
        A for either side
        K for kingside, Q for Q side
        """
        if side == "A":
            # want to know if you can castle on either side
            return self.board[8, WHITE_KSIDE_CASTLE_SHIFT, 0] == 1 or self.board[8, WHITE_QSIDE_CASTLE_SHIFT, 0] == 1
        if side == "K":
            side_index = WHITE_KSIDE_CASTLE_SHIFT
        else: # == "Q"
            side_index = WHITE_QSIDE_CASTLE_SHIFT
        return self.board[8, side_index, 0] == 1
    
    def set_white_no_castle(self, type):
        """
        A: for no castling at all
        K: for no kingside
        Q: for no queenside
        """
        if type == "A":
            self.board[8,WHITE_KSIDE_CASTLE_SHIFT, 0] = 0
            self.board[8,WHITE_QSIDE_CASTLE_SHIFT,0] = 0
        elif type == "K":
                self.board[8, WHITE_KSIDE_CASTLE_SHIFT, 0] = 0 # set the white kside to 0
        else:
            self.board[8,WHITE_QSIDE_CASTLE_SHIFT,0] = 0

    

    def black_can_castle(self, side):
        """
        A for either side
        K for kingside, Q for Q side
        """
        if side == "A":
            # want to know if you can castle on either side
            return self.board[8, BLACK_KSIDE_CASTLE_SHIFT, 0] == 1 or self.board[8, BLACK_QSIDE_CASTLE_SHIFT, 0] == 1
        if side == "K":
            side_index = BLACK_KSIDE_CASTLE_SHIFT
        else:
            side_index = BLACK_QSIDE_CASTLE_SHIFT
        return self.board[8, side_index, 0] == 1
    
    def set_black_no_castle(self, side):
        """
        A: for no castling at all
        K: for no kingside
        Q: for no queenside
        """
        if side == "A":
            self.board[8,BLACK_KSIDE_CASTLE_SHIFT, 0] = 0
            self.board[8,BLACK_QSIDE_CASTLE_SHIFT,0] = 0
        elif side == "K":
            self.board[8, BLACK_KSIDE_CASTLE_SHIFT, 0] = 0 # set the white kside to 0
        else:
            self.board[8,BLACK_QSIDE_CASTLE_SHIFT,0] = 0
