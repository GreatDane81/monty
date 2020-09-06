import numpy as np

import chess

ROWS = 9 # 8 for rows, + 1 for storing heuristics. Add more rows with caution
COLUMNS = 8 # one per col

LETTER_TO_ROW = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4,'f':5, 'g':6,'h':7}

VALUE_TO_PIECE = {128:'k', 9:'q', 5:'r', 4:'b', 3:'n', 1:'p', -128:'K', -9:'Q', -5:'R', -4:'B', -3:'N', -1:'P'}
PIECE_TO_VALUE = {'k':128, 'q':9, 'r':5, 'b':4, 'n':3, 'p':1, 'K':-128, 'Q':-9, 'R':-5, 'B':-4, 'N':-3, 'P':-1} # stored WRT white, mult by BLACK_MULTIPLIER to get black values
WHITE_MULTIPLIER = 1
BLACK_MULTIPLIER = -1
EMPTY_SQUARE = 0

# py chess stuff for promotion
PIECE_TYPE_TO_VALUE = {6:128, 5:9, 4:5, 3:4, 2:3, 1:1}

# Heuristics:
TURN_BIT_SHIFT = 0 # at [8,0]
WHITE_KSIDE_CASTLE_SHIFT = 1 # will be at board[8,1]
WHITE_QSIDE_CASTLE_SHIFT = 2 # will be at board[8,2]
BLACK_KSIDE_CASTLE_SHIFT = 3 # will be at board[8,3]
BLACK_QSIDE_CASTLE_SHIFT = 4 # will be at board[8,4]
MATERIAL_DIFFERENCE_SHIFT = 5 # will be stored at board[8,5]
WHITE_TOTAL_MATERIAL = 6 # [8,6] # TODO: Not sure this is the best heuristic, but I'll try it.
BLACK_TOTAL_MATERIAL = 7 # [8,7]
class TwoDBoard:
    """
    Same idea as before, but this time I'm going to make two dimensional input, not 3. As before, the 9th row [8] is for heureustics,
    and maybe I'll add more.

    The difference is this time instead of a bit map, each square will contain its positive worth if the piece is occupied by white, 
    negative its worth if black, and 0 if empty.

    Kings will be given a weight of 128, and bishops a weight of 4 to differentiate them from knights. The hope is that
    this pattern will be more learnable: If members of the opposite weight relate to the king in a certain way, the score
    is raised/lowered.

    Also generally, the more pieces you have on the board the better off you are.

    A lot of the same machinery for board maintenance will be ported.
    """
    def __init__(self):
        """
        Constructor for standard starting position
        """
        self.board = np.zeros([ROWS, COLUMNS],dtype='float32')
        # fill the 2nd rank with white pawns
        self.board[1, :] = PIECE_TO_VALUE['p']
        # fill the 7th rank with black pawns
        self.board[6, :] = PIECE_TO_VALUE['P']
        self.set_back_rank(WHITE_MULTIPLIER) # set the white back rank
        self.set_back_rank(BLACK_MULTIPLIER) # set the black white rank
        # Done with the board, as everything else is set to 0
        # Heuristics:
        self.board[8,TURN_BIT_SHIFT] = 0 # turn bit, making this explicit
        self.board[8,WHITE_KSIDE_CASTLE_SHIFT] = 1
        self.board[8,WHITE_QSIDE_CASTLE_SHIFT] = 1
        self.board[8,BLACK_KSIDE_CASTLE_SHIFT] = 1
        self.board[8,BLACK_QSIDE_CASTLE_SHIFT] = 1
    
    def set_back_rank(self, multiplier):
        """
        Sets the appropriate back rank
        """
        if multiplier == WHITE_MULTIPLIER:
            row = 0
        else:
            row = 7
        self.board[row, 0] = multiplier*PIECE_TO_VALUE['r']
        self.board[row, 1] = multiplier*PIECE_TO_VALUE['n']
        self.board[row, 2] = multiplier*PIECE_TO_VALUE['b']
        self.board[row, 3] = multiplier*PIECE_TO_VALUE['q']
        self.board[row, 4] = multiplier*PIECE_TO_VALUE['k']
        self.board[row, 5] = multiplier*PIECE_TO_VALUE['b']
        self.board[row, 6] = multiplier*PIECE_TO_VALUE['n']
        self.board[row, 7] = multiplier*PIECE_TO_VALUE['r']
        # I checked, q/k order correct given board coord system
    
    def __str__(self):
        """
        Converts to human readable string

        White pieces are lower case
        """
        # will go rank 8 down to generate a string where white is on the bottom.
        s = ""
        for i in range(7, -1, -1): # starting row 7, ending row 
            s += self.print_row(i) + "\n"
        return s
    
    def print_row(self, row):
        """
        Prints a single row
        """
        s = ""
        for col in range(0, 8):
            if self.board[row, col] == 0:
                # then no piece here
                s += "."
            else: 
                s += TwoDBoard.piece_to_str(self.board, row, col)
        return s
    
    def empty_index(self, row, col):
        """
        Empties the given index
        """
        self.board[row, col] = 0
    
    def update_index(self, row, col, piece):
        """
        Writes over the given index with piece
        """
        self.board[row, col] = piece
    
    def update_turn(self):
        """
        Flips the turn bit at [8,0]
        """
        self.board[8,0] = 1 - self.board[8,0]
    
    def update_material_difference(self):
        """
        Updates all three heurestics: total white, total black, difference
        """
        total_white = 0
        total_black = 0
        for row in range(0,8):
            for col in range(0,8):
                if self.board[row,col] > 0:
                    # white piece
                    total_white += self.board[row,col]
                elif self.board[row, col] < 0:
                    total_black += BLACK_MULTIPLIER*self.board[row,col]
        # after checking the entire board, update bits
        self.board[8,WHITE_TOTAL_MATERIAL]= total_white
        self.board[8,BLACK_TOTAL_MATERIAL]= total_black
        self.board[8,MATERIAL_DIFFERENCE_SHIFT] = total_white - total_black


    def white_can_castle(self, side):
        """
        A for either side
        K for kingside, Q for Q side
        """
        if side == "A":
            # want to know if you can castle on either side
            return self.board[8, WHITE_KSIDE_CASTLE_SHIFT] == 1 or self.board[8, WHITE_QSIDE_CASTLE_SHIFT] == 1
        if side == "K":
            side_index = WHITE_KSIDE_CASTLE_SHIFT
        else: # == "Q"
            side_index = WHITE_QSIDE_CASTLE_SHIFT
        return self.board[8, side_index] == 1
    
    def set_white_no_castle(self, type):
        """
        A: for no castling at all
        K: for no kingside
        Q: for no queenside
        """
        if type == "A":
            self.board[8,WHITE_KSIDE_CASTLE_SHIFT] = 0
            self.board[8,WHITE_QSIDE_CASTLE_SHIFT] = 0
        elif type == "K":
                self.board[8, WHITE_KSIDE_CASTLE_SHIFT] = 0 # set the white kside to 0
        else:
            self.board[8,WHITE_QSIDE_CASTLE_SHIFT] = 0  
    
    def black_can_castle(self, side):
        """
        A for either side
        K for kingside, Q for Q side
        """
        if side == "A":
            # want to know if you can castle on either side
            return self.board[8, BLACK_KSIDE_CASTLE_SHIFT] == 1 or self.board[8, BLACK_QSIDE_CASTLE_SHIFT] == 1
        if side == "K":
            side_index = BLACK_KSIDE_CASTLE_SHIFT
        else:
            side_index = BLACK_QSIDE_CASTLE_SHIFT
        return self.board[8, side_index] == 1
    
    def set_black_no_castle(self, side):
        """
        A: for no castling at all
        K: for no kingside
        Q: for no queenside
        """
        if side == "A":
            self.board[8,BLACK_KSIDE_CASTLE_SHIFT] = 0
            self.board[8,BLACK_QSIDE_CASTLE_SHIFT] = 0
        elif side == "K":
            self.board[8, BLACK_KSIDE_CASTLE_SHIFT] = 0 # set the white kside to 0
        else:
            self.board[8,BLACK_QSIDE_CASTLE_SHIFT] = 0
    
    def push(self, move, promotion=None):
        """
        Pushes move, with promotion.
        Move is a str.
        """
        # start by getting the move indices
        indices = TwoDBoard.parse_move_indices(move)
        start_index = indices[0]
        target_index = indices[1]
        piece = TwoDBoard.get_piece_from_index(self.board, start_index[0], start_index[1])
        if piece == None:
            raise ValueError # more error checking
        if VALUE_TO_PIECE[piece] == "k":
            # then white can no longer castle
            self.set_white_no_castle("A")
        elif VALUE_TO_PIECE[piece] == "K":
            self.set_black_no_castle("A")
        elif piece == "r" and start_index == Board.parse_index("h1"):
            self.set_white_no_castle("K")
        elif piece == "r" and start_index == Board.parse_index("a1"):
            self.set_white_no_castle("Q")
        elif piece == "R" and start_index == Board.parse_index("h8"):
            self.set_black_no_castle("K")
        elif piece == "R" and start_index == Board.parse_index("a8"):
            self.set_black_no_castle("Q")
        self.empty_index(start_index[0], start_index[1])
        # then, update the target index
        if promotion != None:
            # Then get the promotion piece
            piece_type = move.promotion
            multiplier = WHITE_MULTIPLIER
            if self.get_move(): # meaning, it's black's turn
                multiplier = BLACK_MULTIPLIER
            piece = multiplier*PIECE_TYPE_TO_VALUE[piece_type]
        self.update_index(target_index[0], target_index[1], piece)
        # update material count
        self.update_material_difference()
        # And switch player turns
        self.update_turn()

    
    def castle(self, colour, side):
        """
        Castling called on any instance of castling.
    
        Will attempt to castle with no checks, assumes valid castle provided
        """
        if colour == "W":
            rank = "1"
            king_char, rook_char = "k", "r"
            self.set_white_no_castle("A")
        else: # 'B'
            rank = "8"
            king_char, rook_char = "K", "R"
            self.set_black_no_castle("A")
        if side == "K":
            # check the king and rook are in the correct spots
            king_start_index = TwoDBoard.parse_index("e"+rank)
            rook_start_index = TwoDBoard.parse_index("h"+rank)
            king_target_index = TwoDBoard.parse_index("g"+rank)
            rook_target_index = TwoDBoard.parse_index("f"+rank)
            king = PIECE_TO_VALUE[king_char]
            rook = PIECE_TO_VALUE[rook_char]
        else: # 'Q'
            king_start_index = TwoDBoard.parse_index("e"+rank)
            rook_start_index = TwoDBoard.parse_index("a"+rank)
            king_target_index = TwoDBoard.parse_index("c"+rank)
            rook_target_index = TwoDBoard.parse_index("d"+rank)
            king = PIECE_TO_VALUE[king_char]
            rook = PIECE_TO_VALUE[rook_char]
        # Now check the pieces are in the right place to start
        alleged_king =  TwoDBoard.get_piece_from_index(self.board, king_start_index[0], king_start_index[1])
        king_correct = alleged_king == king
        alleged_rook = TwoDBoard.get_piece_from_index(self.board, rook_start_index[0], rook_start_index[1])
        rook_correct =  alleged_rook == rook
        if not king_correct or not rook_correct:
            print(self.board)
            print(self)
            raise ValueError # Raise an error
        # next check the squares being castled to aren't occupied
        if TwoDBoard.get_piece_from_index(self.board, king_target_index[0], king_target_index[1]) != EMPTY_SQUARE:
            raise ValueError
        if TwoDBoard.get_piece_from_index(self.board, rook_target_index[0], rook_target_index[1]) != EMPTY_SQUARE:
            raise ValueError
        # now we can empty the old indices and update the new ones
        self.empty_index(king_start_index[0], king_start_index[1])
        self.empty_index(rook_start_index[0], rook_start_index[1])
        # update
        self.update_index(king_target_index[0], king_target_index[1], king)
        self.update_index(rook_target_index[0], rook_target_index[1], rook)

    def get_move(self):
        return self.board[8, TURN_BIT_SHIFT]



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
        start_index = TwoDBoard.parse_index(start_sq)
        target_index = TwoDBoard.parse_index(target_sq)
        return [start_index, target_index]
    
    @staticmethod
    def parse_index(index):
        '''
        "a1" --> (0, 0)
        '''
        # TODO: Assert length = 2
        return (int(index[1]) - 1, LETTER_TO_ROW[index[0]])
    
    @staticmethod
    def get_piece_from_index(board, row, col):
        '''
        (row, col) -> float

        A1 -> 0, 0
        '''
        return board[row, col]

    @staticmethod
    def piece_to_str(board, row, col):
        if board[row, col] == 0:
            return None # error checking
        return VALUE_TO_PIECE[board[row, col]]

