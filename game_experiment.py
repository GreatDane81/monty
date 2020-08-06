import chess
import chess.pgn

import chess.engine
from chess.engine import Cp, Mate, MateGiven

from ai_experiment import MoveNode, MoveSelector

def play(d):
    """
    play with tree depth = d, from start
    """
    # make the board
    board = chess.Board()
    # Make the root decision node
    player = 0
    root = MoveNode(board, player = player, depth = d, move = None)
    i = 0
    while i < 3:
        result = MoveSelector.pick_move(root)
        move = result[1].move
        board.push(move)
        player = 1 - player
        root = MoveNode(board, player = player, depth = d, move = None)
        print(move)
        print(board, "\n")
        i += 1
x = 2
play(x) # It's playing!!
board = chess.Board()
# Make the root decision node
#player = 0
#root = MoveNode(board, player = player, depth = d, move = None)
#print(root)

# NOTES:
# Think I should get rid of the tree shit. Slows down the computation magnificiently and won't add
# much to the point of the project. Maybe I can add it back in if Keras is faster, but I doubt it will be.