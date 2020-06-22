import chess
import chess.pgn

import numpy as np

import tensorflow as tf

import copy

import chess.engine


SF_DEPTH = 10 # depth for stockfish

class MoveNode:

    def __init__(self, board, player, depth):
        '''
        Must be python chess board for 'legal moves' functionality + evaluation
        '''
        self.board = board
        self.value = None # Will catch errors, don't tell anyone though
        self.children = [] # list for easy traversal. This will be a list of MoveNodes
        self.player = player # 0 for white, 1 for black
        self.depth = depth # Will be used for evaluation
        self.build_structure(depth) # And build the structure
    
    def build_structure(self, depth):
        '''
        0 for just this position
        1 for one level of moves
        '''
        self.depth = depth
        if depth == 0:
            return
        new_player = 1 - self.player
        for move in self.board.legal_moves:
            # create a new board and push... ugh the memory this might get ugly.
            board.push(move)
            new_board = copy.deepcopy(board) # oh boy, this is really not good
            board.pop() # man this will get real bad for memory, not even sure this is viable
            new_node = MoveNode(new_board, new_player, depth - 1)
            self.children.append(new_node)
    
class MoveSelector:
    # ohboy.jpg is the memory going to be dog

    def __init__(self, engine):
        '''
        Eventually the engine will be my Keras model.
        For now we use stockfish directly
        '''
        self.engine = engine

    @staticmethod
    def pick_move(root):
        '''
        root is a MoveNode
        '''
        if root.depth == 0:
            # Then we just take the min/max best move
            if root.player == 0:
                # trying to maximize
                first_board = root.children[0].board
                cur_val = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH))

board = chess.Board()
root = MoveNode(board, 0, 0)

root.build_structure(1) # Yea man i mean this memory useage will be pretty heavy but i don't see
# a work around.

print(root.board)

# So the goal here is to make a nice traversable tree for min/max

file_path = 'C:/Users/Ethan Dain/Desktop/University/Machine Learning/Code/monty/kasparov-deep-blue-1997.pgn'
file = open(file_path)

first_game = chess.pgn.read_game(file)
board = first_game.board()

# Ok, so figured out how to generate legal moves from any given position
#for move in first_game.mainline_moves():
#    board.push(move)
#    print(board.legal_moves)