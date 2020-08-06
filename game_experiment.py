import chess
import chess.pgn

import chess.engine
from chess.engine import Cp, Mate, MateGiven

from ai_experiment import MoveNode, MoveSelector

def play():
    # make the board
    board = chess.Board()
    # Make the root decision node
    player = 0
    root = MoveNode(board, player = player, depth = 1, move = None)
    i = 0
    while i < 3:
        result = MoveSelector.pick_move(root)
        move = result[1].move
        board.push(move)
        player = 1 - player
        root = MoveNode(board, player = player, depth = 1, move = None)
        print(move)
        print(board, "\n")
        i += 1

play() # It's playing!!
