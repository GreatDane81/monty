import chess
import chess.pgn


import numpy as np

import tensorflow as tf

import copy

import chess.engine
from chess.engine import Cp, Mate, MateGiven

from tensorflow import keras




SF_DEPTH = 10 # depth for stockfish

ANALYSIS_TIME = 0.1 # in seconds

#path = "C:/Users/Ethan Dain/Desktop/University/Machine Learning/Code/monty/stockfish/stockfish-11-win/stockfish-11-win/Windows/stockfish_20011801_x64.exe"

engine = keras.models.load_model('conv_model.h5')

class MoveNode:

    def __init__(self, board, player, depth, move):
        '''
        Must be python chess board for 'legal moves' functionality + evaluation
        '''
        self.my_board = board
        self.game_board = board.pychess_board
        #print(self.game_board)
        self.move = move # The move represented by this node
        self.children = [] # list for easy traversal. This will be a list of MoveNodes
        self.player = player # 0 for white, 1 for black
        self.depth = depth # Will be used for evaluation, = 0 when at base of tree
        self.build_structure(depth) # And build the structure
    
    def build_structure(self, depth):
        '''
        0 for just this position
        1 for one level of moves
        '''
        if depth == 0:
            return # no tree to build, just return the move
        for move in self.game_board.legal_moves:
            # create a new board and push... ugh the memory this might get ugly.
            #print("before playing the move:")
            #print(self.my_board)
            self.my_board.play_move(move)
            new_board = copy.deepcopy(self.my_board) # oh boy, this is really not good
            #print("the board that was copied")
            #print(new_board)
            #print(new_board)
            #print("\n\n")
            self.my_board.pop() # man this will get real bad for memory, not even sure this is viable
            #print("The board that was restored after pop:")
            #print(self.my_board)
            new_node = MoveNode(new_board, player = 1 - self.player, depth = depth - 1, move = move)
            self.children.append(new_node)
    
    def __str__(self):
        '''
        printing because debug is slow
        '''
        # print layer by layer:
        s = "\ndepth = " +  (str)(self.depth) +  ", move = " +  (str)(self.move) + ", player = " + (str)(self.player)
        for child in self.children:
            s += child.__str__()
        return s
    
class MoveSelector:
    # ohboy.jpg is the memory going to be dog

    def __init__(self, engine):
        '''
        Eventually the engine will be my Keras model.
        For now we use stockfish directly
        '''
        self.engine = engine # TODO: Make this global or mutable for the entire class at once
        
    @staticmethod
    def pick_move(root):
        '''
        root is a MoveNode
        Pick a move that is best to play for self.player
        '''
        # Two cases:
        # Either the "root" is a leaf, i.e depth = 0
        if root.depth == 0:
            # Here there is no decision to make, just return the move and the engine's evaluation
            #score =  engine.analyse(root.board, chess.engine.Limit(time=ANALYSIS_TIME))["score"]
            root_board = root.my_board # but since this is my object,
            tensor_board = root_board.board
            score = engine.predict(np.array([tensor_board,]))
            #print(root_board)
            return (score, root)
        else:
            # The "root" is somewhere else in the tree, in which case it will choose the best child.
            # resolve all the children, and take the best move
            opt_child = root.children[0] # know it has kids because depth is non-0
            opt_score = MoveSelector.pick_move(opt_child)[0]
            for i in range(1, len(root.children)):
                child = root.children[i]
                result = MoveSelector.pick_move(child)
                if MoveSelector.compare_scores(result[0], opt_score, root.player):
                    opt_score = result[0]
                    opt_child = result[1]
            return (opt_score, opt_child)

            
    @staticmethod
    def compare_scores(first, second, player):
        '''
        Returns True if and only if 'first' is a score better for 'player' than 'second' is
        '''
        return first > second

# Ok, so figured out how to generate legal moves from any given position
#for move in first_game.mainline_moves():
#    board.push(move)
#    print(board.legal_moves)