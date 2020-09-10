# for pychess
import chess
from chess.engine import Cp, Mate, MateGiven
import chess.pgn

# for keras
from tensorflow import keras

# for numpy
import numpy as np

# for copy
import copy

# for my stuff
import Board
from ai_experiment import MoveNode, MoveSelector

def select_move(py_board, game_board):
    '''
    Looks at all legal moves, picks best one

    Currently for black
    '''
    legal_moves = game_board.legal_moves
    best_score = 1000000 # just some really high number for white leading
    best_move = None
    save_board = copy.deepcopy(py_board)
    print("legal moves: ", legal_moves)
    for move in legal_moves:
        py_board.play_move(move)
        my_board = py_board.board
        # get the score
        current_score = model.predict(np.array([my_board,]))
        if current_score < best_score:
            best_score = current_score
            best_move = move
        # and reset the py_board
        py_board = copy.deepcopy(save_board)
    return (best_move, best_score, py_board)

def select_move_tree(py_board, game_board, depth):
    """
    Makes a move node of the given depth, plays accordingly

    With depth = 1, this will push all the legal moves onto the board,
    see what's white immediate reply is for each of those, and take the best (lowest) score
    """
    #print("game_board == py_board.pychess_board",game_board == py_board.pychess_board)
    #print("game_board legal moves:")
    #for move in game_board.legal_moves:
    #    print(move, end=" ")
    #print("pychess legal moves:")
    #for move in py_board.pychess_board.legal_moves:
    #    print(move, end=" ")
    copy_board = copy.deepcopy(py_board)
    opt_score = 100000 # some really bad thing for black
    opt_move = None
    for move in game_board.legal_moves:
        #print(move)
        # push the move onto the given board
        #print("board before push:")
        #print(py_board.pychess_board)
        #py_board.play_move(move)
        #print("board after push:")
        #print(py_board.pychess_board)
        #new_board = copy.deepcopy(py_board)
        root = MoveNode(copy_board, py_board.get_turn(), depth, move)
        # and immediately pop it
        #py_board.pop()
        #print("board restored:")
        #print(py_board.pychess_board)
        chosen_move_score, chosen_move = MoveSelector.pick_move(root)
        chosen_move_extract = chosen_move_score[0]
        if chosen_move_score < opt_score:
            # then black is doing better
            opt_move = chosen_move
            opt_score = chosen_move_score
    return (opt_move.move, opt_score)



model = keras.models.load_model("conv_model.h5")

game_board = chess.Board()


#print("before result")
#print(python_board)
#result = select_move(python_board, game_board)
#monty_move = result[0]
#python_board = result[2]
#print("when taken from results")
#print(python_board)

#python_board_2 = Board.Board()
#python_board_2.play_move('e2e4', None)
#python_board_2.play_move('g8h6', None) # somehow this move is pushed but
# i can't for the life of me figure out why
# because the board before being returned looks good, then after being returned
# it's messed up.
#s=str(python_board)
#s1= str(python_board_2)
#print(s == s1)
#move = result[0]
#move_str = str(move)

# TODO: OK apparently the board gets corrupted somehow, somewhere if you don't return it's copy?
# even though without returning its copy it looks totally fine? definitely something to look into.
# weirdest bug i've seen in my life.

my_board = Board.Board()

print("Welcome! You open with the white pieces.")
while True:
    user_input = input("Enter your move now:")
    user_move = chess.Move.from_uci(user_input)
    # push the move on the python board and the game board
    try:
        my_board.play_move(user_move)
        game_board.push(user_move)
    except:
        print("Illegal move, try again")
        user_input = input("Enter your move now:")
        my_board.play_move(user_move)
        game_board.push(user_move)
    # predict the best move
    print("Monty is thinking...")
    result = select_move_tree(my_board, game_board, 2)
    # after getting the result, push the move and update the board
    monty_move = result[0]
    print("Monty replied with:", monty_move)
    my_board.play_move(monty_move)
    game_board.push(monty_move)
    #print(my_board)

# Working on depth 0, but there's a problem between linkup of pychess board and my board. Getting some none type errors.

# don't want separation of gameboard from pyboard, it gives internal problems. 
#  using "game_board" as a temporary fix to get all the legal moves.

# my board is at least being restored correctly, and the print of the pychess board suggests it is also being restored correctly.
# and the legal moves are the same, so we're good in that respect.

# Ok so: 1. e2e4 (M)e7e6 2. d2d4 (M)b8a6

# better depth will give me better results, the question is 1.) how to do it without so many deep copies

# I think I want to rewrite the tree structure, because it's broken right now. Need it to be such that:
# depth 0 just returns the score of the given position
# depth 1 creates a list of children (m)

# Ok figured out the current bug. Pick move returns the best node for the CURRENT player, and it returns the MOVE to make
# for a different depth. The solution really is rewriting the tree structure for
# 1) memory efficiency (where possible, ok fine i know it's a bad exp)
# 2) consistency (because what i have in ai_experiment is not worth salvaging outside the idea.)

# What i've noticed about the network's play: it's trying to "get out of the way" of attacks, and will attack hanging pieces.
# this was all on depth 0. If you make a proper tree structure, you'll see that "ok if i attack this hanging piece it's possible
# for him to attack me back, then I lose a piece. Therefore, he would play that and I would lose. So I won't attack the piece to begin with."

# A good understanding of material. Seems to value knight moves and piece activation, which is good. Some weird rook moves. This network has potential,
# and i can ultimately improve on it in two ways. 1) lowering the loss. This is key because if my "position evaluations" are awful, there's no point continuing with
# a network that's guessing and 2) making a good tree structure that allows me to go 5 or 6 levels deep. Hopefully. Look up alpha beta pruning.

