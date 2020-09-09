# for np
import numpy as np

# for attack stuff
import chess
from chess.engine import Cp, Mate, MateGiven
import chess.pgn

ROWS = 8
COLUMNS = 8
LAYERS = 9 # 9: 6 for piece structure (one layer for each peice, 1/-1 for black/white), 2 for SQUARES_ATTACKED and HEURESTICS

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

# HEURESTICS LAYER
HEURESTICS_LAYER = 8
# HEURESTICS BIT SHIFTS
TURN_INDEX = [0,0] # going to leave the rest of the row blank for pattern recognition
WHITE_K_SIDE_CASTLE_INDEX = [2,0] # has to be two dimensions becuase hereustics is now a layer
WHITE_Q_SIDE_CASTLE_INDEX = [2,1]
BLACK_K_SIDE_CASTLE_INDEX = [2,2]
BLACK_Q_SIDE_CASTLE_INDEX = [2,3]
MATERIAL_DIFFERENCE_INDEX = [4, 0]
WHITE_TOTAL_MATERIAL_INDEX = [4,1]
BLACK_TOTAL_MATERIAL_INDEX = [4,2]

# PRESENT BITS
WHITE_PRESENT = 1
BLACK_PRESENT = -1

# LAYER TO STR
WHITE_LAYER_TO_PIECE = {KING_LAYER:'k', QUEEN_LAYER: 'q', ROOK_LAYER:'r', BISHOP_LAYER:'b', KNIGHT_LAYER: 'n', PAWN_LAYER: 'p'}
BLACK_LAYER_TO_PIECE = {KING_LAYER:'K', QUEEN_LAYER: 'Q', ROOK_LAYER:'R', BISHOP_LAYER:'B', KNIGHT_LAYER: 'N', PAWN_LAYER: 'P'}

# PIECE TO LAYER (all in one dict for useability in update_index)
PIECE_TO_LAYER = {'k': KING_LAYER, 'q': QUEEN_LAYER, 'r': ROOK_LAYER, 'b': BISHOP_LAYER, 'n': KNIGHT_LAYER, 'p': PAWN_LAYER,
                  'K': KING_LAYER, 'Q': QUEEN_LAYER, 'R': ROOK_LAYER, 'B': BISHOP_LAYER, 'N': KNIGHT_LAYER, 'P': PAWN_LAYER}

# MATERIAL VALUE, wrt to absolute value
MATERIAL_VALUE = {'p': 1, 'b':3, 'n':3, 'r':5, 'q':9} # intentionally omitted king
                                                                                                                            
# For promotion and conversion of types:
PYCHESS_PIECE_TYPE_TO_BOARD_PIECE_TYPE = {1:'p', 2:'n', 3:'b', 4:'r', 5:'q', 6:'k'}

# n for knight for disambuity, I'm sorry
LETTER_TO_ROW = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4,'f':5, 'g':6,'h':7}


class Board: 
    # TODO: IMPORTANT: call order in play_experiment should be 1. Board.play, 2. Board. pychessboard.play, for computation of attack layers
    def __init__(self):
        '''
        Creates a new board with standard starting position
        '''
        # start with an empty board
        self.board = np.zeros([ROWS, COLUMNS, LAYERS],dtype='float32')
        # adding a PyChess board to each Board for generating attack layers
        self.pychess_board = chess.Board() 
        # set up piece layers
        pieces_to_index = {'k':{'e1'}, 'q':{'d1'}, 'r':{'a1','h1'}, 'n':{'b1', 'g1'}, 'b':{'f1','c1'}, 'p':{'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'},
                           'K':{'e8'}, 'Q':{'d8'}, 'R':{'a8','h8'}, 'N':{'b8', 'g8'}, 'B':{'f8','c8'}, 'P':{'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7'}}
        self.pieces_setup(pieces_to_index)
        # set up attack layers
        self.attack_layer_setup()
        # set up heurestics layer
        self.heurestics_setup()

    def pieces_setup(self, pieces_to_index):
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
    
    def heurestics_setup(self):
        """
        Sets up the heurestics row

        Which has material difference, total material for each side, castling rights, turn
        """
        # start by explicitly setting the turn bit
        self.board[TURN_INDEX[0], TURN_INDEX[1], HEURESTICS_LAYER] = 0
        # Then initialize the castling rights
        self.board[WHITE_K_SIDE_CASTLE_INDEX[0], WHITE_K_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 1
        self.board[WHITE_Q_SIDE_CASTLE_INDEX[0], WHITE_Q_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 1
        self.board[BLACK_K_SIDE_CASTLE_INDEX[0], BLACK_K_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 1
        self.board[BLACK_Q_SIDE_CASTLE_INDEX[0], BLACK_Q_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 1
        # Then do material, will be updated on the fly. Promotions will be handed.
        self.board[MATERIAL_DIFFERENCE_INDEX[0], MATERIAL_DIFFERENCE_INDEX[1], HEURESTICS_LAYER] = 0 # start even
        self.board[WHITE_TOTAL_MATERIAL_INDEX[0], WHITE_TOTAL_MATERIAL_INDEX[1], HEURESTICS_LAYER] = 45 # just the sum
        self.board[BLACK_TOTAL_MATERIAL_INDEX[0], BLACK_TOTAL_MATERIAL_INDEX[1], HEURESTICS_LAYER] = 45
    

    def attack_layer_setup(self):
        """
        Sets up both attack layers, called after every move
        """
        # start by reseting both layers:
        self.board[:, :, WHITE_ATTACK_LAYER] = np.zeros([ROWS, COLUMNS], dtype='float32')
        self.board[:, :, BLACK_ATTACK_LAYER] = np.zeros([ROWS, COLUMNS], dtype='float32')
        for square in chess.SQUARES:
            # get the attack set
            attack_set = self.pychess_board.attacks(square)
            if self.pychess_board.color_at(square) == chess.WHITE:
                layer = WHITE_ATTACK_LAYER
            elif self.pychess_board.color_at(square) == chess.BLACK:
                layer = BLACK_ATTACK_LAYER
            else:
                layer = None
            if layer != None:
                # then want to go through all the squares to find attacks
                for attacked_sq in attack_set:
                    # get the index (row, col)
                    my_index = Board.pychess_sq_to_my_sq(attacked_sq)
                    # and increment the right counter
                    self.board[my_index[0], my_index[1], layer] += 1


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

    def play_move(self, move):
        '''
        (pychess "Move") -> None

        Updates the board given coordinates for a move, promotes if necessary.
        '''
        # start by getting indices
        move_str = str(move)
        indices = Board.parse_move_indices(move_str)
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
        # then, update the target index
        if move.promotion != None:
            # Then get the promotion piece
            py_chess_piece_type = move.promotion
            piece = PYCHESS_PIECE_TYPE_TO_BOARD_PIECE_TYPE[py_chess_piece_type]
            upgrade_result = MATERIAL_VALUE[piece] - MATERIAL_VALUE['p']
            if self.get_turn() == WHITES_TURN:
                # Update material:
                self.update_white_material(upgrade_result) 
                # INCREASE the material difference
                self.update_material_difference(upgrade_result)
            else:
                # increase the total worth of black's material
                self.update_black_material(upgrade_result)
                # decrease the material difference, because this is better for black
                upgrade_result *= -1
                self.update_material_difference(upgrade_result)
                # then take it to uppercase for updating correcting
                piece = piece.upper()
        self.update_index(target_index[0], target_index[1], piece)
        # now it's safe to remove the old index
        self.empty_index(start_index[0], start_index[1])
        # then push the move onto  the board
        self.pychess_board.push(move)
        # update the attack indices
        self.attack_layer_setup()
        # And switch player turns
        self.update_turn()


    def empty_index(self, row, col):
        '''
        Empties the square at [row, col].

        Updates material accordingly.
        '''
        # get the piece layer
        piece_layer = Board.get_piece_layer_from_index(self.board, row, col)
        if piece_layer == None:
            raise ValueError("Tried to empty and empty square")
        # update the layer to 0
        self.board[row, col, piece_layer] = 0
        

    def update_index(self, row, col, piece):
        '''
        (int, int, char)
        Writes over the square at row, col, with 'piece'.
        Note: Old piece is wiped, this accounts for captures.
        '''
        # First, check if there was occupying the new square
        old_piece = Board.get_piece_from_index(self.board, row, col)
        if old_piece != None:
            # then this move is a capture and we need update material
            # get the old piece's value
            if old_piece.islower():
                # then it's a white piece that was captured
                value = MATERIAL_VALUE[old_piece]
                # lower white's total material
                loss = -1*value
                self.update_white_material(loss)
                # and decrease the material
                self.update_material_difference(loss)
            else:
                # it's a black piece that was captured
                value = MATERIAL_VALUE[old_piece.lower()]
                loss = -1*value
                self.update_black_material(loss)
                # and increase the score, because white is better
                self.update_material_difference(value)
            # And now you can set the old layer to 0
            self.board[row, col, PIECE_TO_LAYER[old_piece]] = 0
        # next, find the appropriate layer to put in the new piece
        layer = PIECE_TO_LAYER[piece]
        if piece.islower():
            # then update with WHITE_PRESENT
            self.board[row, col, layer] = WHITE_PRESENT
        else:
            self.board[row, col, layer] = BLACK_PRESENT


    def castle_king_side(self, colour,move):
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
        self.castling_logic(king_index, rook_index, king_target, rook_target, king_char, rook_char,move)
        

    def castle_queen_side(self, colour,move):
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
        self.castling_logic(king_index, rook_index, king_target, rook_target, king_char, rook_char, move)

    def castling_logic(self, king_index, rook_index, king_target, rook_target, king_char, rook_char, move):
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
        # conclude by updating the turn, pushing onto the pychess board, and updating attack layers
        self.pychess_board.push(move)
        self.attack_layer_setup()
        self.update_turn()

    def update_turn(self):
        """
        Flips the turn bit
        """
        turn = self.board[TURN_INDEX[0],TURN_INDEX[1], HEURESTICS_LAYER]
        self.board[TURN_INDEX[0],TURN_INDEX[1], HEURESTICS_LAYER] = 1 - turn
    
    def get_turn(self):
        return self.board[TURN_INDEX[0],TURN_INDEX[1], HEURESTICS_LAYER]

    def white_can_castle(self, side, move):
        """
        (char) -> None

        A for either side
        K for kingside, Q for Q side
        """
        kside = self.board[WHITE_K_SIDE_CASTLE_INDEX[0], WHITE_K_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] == 1
        qside = self.board[WHITE_Q_SIDE_CASTLE_INDEX[0], WHITE_Q_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] == 1
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
            self.board[WHITE_K_SIDE_CASTLE_INDEX[0], WHITE_K_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 0
            self.board[WHITE_Q_SIDE_CASTLE_INDEX[0], WHITE_Q_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 0
        elif type == "K":
                self.board[WHITE_K_SIDE_CASTLE_INDEX[0], WHITE_K_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 0 # set the white kside to 0
        else:
            self.board[WHITE_Q_SIDE_CASTLE_INDEX[0],WHITE_Q_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 0

    def black_can_castle(self, side):
        """
        (char) -> None

        A for either side
        K for kingside, Q for Q side
        """
        kside = self.board[BLACK_K_SIDE_CASTLE_INDEX[0], BLACK_K_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] == 1
        qside = self.board[BLACK_Q_SIDE_CASTLE_INDEX[0], BLACK_Q_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] == 1
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
            self.board[BLACK_K_SIDE_CASTLE_INDEX[0], BLACK_K_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 0
            self.board[BLACK_K_SIDE_CASTLE_INDEX[0], BLACK_K_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 0
        elif type == "K":
                self.board[BLACK_K_SIDE_CASTLE_INDEX[0], BLACK_K_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 0 # set the white kside to 0
        else:
            self.board[BLACK_K_SIDE_CASTLE_INDEX[0],BLACK_K_SIDE_CASTLE_INDEX[1], HEURESTICS_LAYER] = 0
    
    def update_material_difference(self, amount):
        """
        Will ADD the amount, may be negative if black improved or white lost a piece
        """
        self.board[MATERIAL_DIFFERENCE_INDEX[0], MATERIAL_DIFFERENCE_INDEX[1], HEURESTICS_LAYER] += amount
    
    def update_white_material(self, amount):
        """
        Will ADD the amount, may be negative if lost ground
        """
        self.board[WHITE_TOTAL_MATERIAL_INDEX[0], WHITE_TOTAL_MATERIAL_INDEX[1], HEURESTICS_LAYER] += amount
    
    def update_black_material(self, amount):
        """
        Will ADD the amount, may be negative if lost ground
        """
        self.board[BLACK_TOTAL_MATERIAL_INDEX[0], BLACK_TOTAL_MATERIAL_INDEX[1], HEURESTICS_LAYER] += amount
        
    
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
            if board[row, col, layer] != 0:
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
        piece = board[row, col, piece_layer]
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

    @staticmethod
    def pychess_sq_to_my_sq(pychess_sq):
        """
        Have to do this translation by hand, because library code
        not working for me
        """
        # totally took this from the source
        row = pychess_sq >> 3
        col = pychess_sq & 7
        return (row, col)