# My files: the board tensor and file/training stuff
import BoardTensor
import paths
import training_limits

# py chess stuff
import chess
from chess.engine import Cp, Mate, MateGiven
import chess.pgn

# training array + pickle for file storage, np used for serialization too
import numpy as np
import pickle

# for sigmoid
import math


# Engine configuration: 
SF_DEPTH = 10 # depth for stockfish
ANALYSIS_TIME = 0.1 # in seconds
engine = chess.engine.SimpleEngine.popen_uci(paths.ENGINE_PATH)

def generate_train_data_from_PGN(pgn, out_file_path, seen_positions_path, limit):
    """
    (file, str, str, int) -> None

    'pgn' is a file containing a PGN
    'out_file_path' will be used for serializing data to the training out file. Will be appended to existing data
    'seen_positions_file' will be used to make sure no position is analyzed twice
    'limit' determines the maximum number of games to analyze from pgn

    Writes the training data generated by the given PGN. 
        - Each move generates a pair in a np array: [8*8*7 numpy board. stockfish score]
    """
    # Start by creating the object that will be serialized
    train_list = load_training_data(out_file_path)
    # next, collect the set from the seen positions
    seen_positions = get_seen_positions(seen_positions_path)
    game_num = 0
    # Get the game from pgn
    game = chess.pgn.read_game(pgn)
    while game != None and game_num < limit:
        # Generate the tensorboard and py boards
        py_board = chess.Board()    
        for move in game.mainline_moves():
            # update the py board
            py_board.push(move) # TODO: this line changes when changing data type
            # but since we want to sanitize our data, we need to check it's not already been seen
            fen =  py_board.fen()
            if fen in seen_positions:
                # skip, don't analyze this position again
                continue
            else:
                # we do need to analyze the position, so add it to the fen so it won't be analyzed again
                seen_positions.add(fen)
            tensor_board = BoardTensor.board_to_tensor(py_board)
            score =  engine.analyse(py_board, chess.engine.Limit(time=ANALYSIS_TIME))["score"].relative
            # convert the score to something useable by the model
            numerical_score = get_numerical_score(score, py_board)
            # and append. Unfortunate that this is O(n)
            # TODO: Find a better way, hopefully this isn't too ugly
            train_list.append((tensor_board, numerical_score))
            print(BoardTensor.print_tensor(tensor_board), numerical_score, score)
            print("--------")
        # finish the game, so increase counter
        game = chess.pgn.read_game(pgn)
        print("finished game", game_num)
        game_num += 1
    # and to clean up, store the training list, update the seen positions
    store_training_data(train_list, out_file_path)
    store_seen_positions(seen_positions_path, seen_positions)
    print("finished writing")

def generate_balanced_data(pgn, out_file_path, limit):
    '''
    Same Idea as before, but here the probability of a score being added to the training list
    will depend on how "average" it is.

    To encourage a balanced data set, the more outlying a score is, the more likely is to be included.

    This was computed given the pandas functionality from Model.py
    '''
    pass

def store_training_data(train_list, out_file_path):
    """
    Saves the 'train_list' to the 'out_file_path' using np.save
    """
    with open(out_file_path, 'wb') as f:
        pickle.dump(train_list, f, pickle.HIGHEST_PROTOCOL)

def load_training_data(out_file_path):
    """
    Loads the 'train_list' from 'outfile_path' and returns it as a useable training list
    """
    with open(out_file_path, 'rb') as of:
        train_list = pickle.load(of)
    return train_list

def erase_train_data(out_file_path):
    """
    WARNING: erases all the data in the file in 'out_file_path', sets an empty list
    """
    with open(out_file_path, 'wb') as f:
        pickle.dump([], f, pickle.HIGHEST_PROTOCOL)


def get_seen_positions(seen_positions_path):
    """
    (str) -> set

    Returns a set containing the FENs for each position seen
    """
    with open(seen_positions_path, 'rb') as f:
        return pickle.load(f)

def store_seen_positions(seen_positions_path, seen_positions):
    """
    (str, set) -> None

    Stores the 'seen_positions' into seen_positions_path
    """
    with open(seen_positions_path, 'wb') as f:
        pickle.dump(seen_positions, f, pickle.HIGHEST_PROTOCOL)

def reset_seen_positions(seen_positions_path):
    """
    WARNING: resets which positions have been seen, pickles an empty set
    """
    with open(seen_positions_path, 'wb') as f:
        pickle.dump(set(), f, pickle.HIGHEST_PROTOCOL)

def get_numerical_score(score, board):
    """
    (PovScore) --> +int if white leading, -int if black leading, 0 if even

    If mate() or mate_given(), return a signal to skip terminate the game
    """
    # TODO: Check one of the branches for the latest version of this
    # since white is positive, will be negative if black is leading, so
    if score.is_mate():
        # figure out which player is winning
        if score.mate() < 0:
            # white is getting mated
            return training_limits.BLACK_MATE_SCORE
        elif score.mate() > 0:
            return training_limits.WHITE_MATE_SCORE
        else:
            # because of a quirk in the library, mate(0) conflates a win and a loss
            # so  I need to find out on my own who won
            if board.turn == chess.WHITE:
                # then white got mated
                return training_limits.BLACK_MATE_SCORE
            else:
                return training_limits.WHITE_MATE_SCORE
    # otherwise we need to do sigmoid on the score
    numerical_score = float(score.score()) 
    numerical_score /= 100 # because it's in centipawn
    return reduce(numerical_score)

def reduce(P):
    """
    float -> (returns white win percentage)
    Hopefully this is numerically sound.

    Reduces the centipawn score to a more digestible training value.

    Input must be in P, not centi P.
    Source: https://www.chessprogramming.org/Pawn_Advantage,_Win_Percentage,_and_Elo
    """
    # seems odd but checkout the graph from above
    return 1/ (1+math.pow(10, (-P/4)))



if __name__ == "__main__":
    reset_seen_positions(paths.SEEN_POSITIONS_PATH)
    morphy_file, morphy_limit = open(paths.PREFIX + "/Morphy.pgn"), training_limits.SIZES['Morphy'] 
    generate_train_data_from_PGN(morphy_file, paths.OUTFILE_PATH, paths.SEEN_POSITIONS_PATH, 1)
    print("generated")
    

