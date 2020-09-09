import Board # had to change python.linting.pylintEnable to false in settings.
# weird, looks like a pyLint thing.
from Game import Game

import chess
from chess.engine import Cp, Mate, MateGiven
import chess.pgn

import pickle
# recall the idea here was:
# - Start the game in an initial state
# - run through the game, and for each move get stockfish's evaluation
# - store a pair: (tensor rep of board, stockfish evaluation) for training


# Want to do this in two ways: 

# 1. By pgn of a game that was played
def generate_train_data_from_PGN(pgn, out_file_path, limit, skip = 0):
    """
    (file, str, int, int -> None

    'pgn' is a file containing a PGN
    'out_file' will be used for pickling data to the training out file. Will be appended to existing data

    Writes the training data generated by the given PGN. 
        - Each move generates a pair: (a list representing the NumpyBoard, stockfish score)
    
    The file that will be serialized will contain one object: a list of every training pair
    """
    # Start by creating the object that will be pickled
    train_list = []
    game_num = 0
    # Get the game from pgn
    fail_count = 0
    positions_count = 0
    board_differential_count = 0
    game = chess.pgn.read_game(pgn)
    game_num += 1
    while game != None and game_num < limit:
        #print(game.headers["White"])
        #print(game.headers["Black"])
        #print(game.headers["Site"])
        #print(game.headers["Date"])
        vanilla_board = chess.Board()
        positions_count = 0
        # Generate the tensorboard and py boards
        np_board =  Board.Board()
        for move in game.mainline_moves():
            # update the np board
            Game.play_move_on_np_board(np_board, move)
            #print(move)
            #print(np_board)
            vanilla_board.push(move)
            if vanilla_board != np_board.pychess_board:
                board_differential_count += 1
            #print("boards are the same:", vanilla_board == np_board.pychess_board)
            try:
                score =  engine.analyse(np_board.pychess_board, chess.engine.Limit(time=ANALYSIS_TIME))["score"]
                #vanilla_score = engine.analyse(vanilla_board, chess.engine.Limit(time=ANALYSIS_TIME))["score"]
                numerical_score = get_numerical_score(score)
                positions_count += 1
                train_list.append((np_board.board, numerical_score))
            except:
                # Sometimes the engine analysis crashes, and I'm confident it's not my own code
                # because I can run through every position in Tal.pgn no worries.
                fail_count += 1
                break # skip the rest of the game
        game = chess.pgn.read_game(pgn)
        print("finished game", game_num, "; positions from game; ", positions_count, "; fails;", fail_count, "; board ineqs;", board_differential_count)
        game_num += 1
    store_training_data(train_list, out_file_path)
    print("finished writing, total positions evaluated:", len(train_list))

def generate_balanced_data(pgn, out_file_path, limit):
    '''
    Same Idea as before, but here the probability of a score being added to the training list
    will depend on how "average" it is.

    To encourage a balanced data set, the more outlying a score is, the more likely is to be included.

    This was computed given the pandas functionality from Model.py
    '''

def store_training_data(train_list, out_file_path):
    """
    [] of (board, score) to be pickled in outfile
    """
    out_file = open(out_file_path, "ab")
    pickle.dump(train_list, out_file)
    out_file.close()

def load_training_data(out_file_path):
    """
    (str) -> training list of (np_board, score)
    """
    out_file = open(out_file_path, "rb")
    train_list = pickle.load(out_file)
    out_file.close()
    return train_list

def erase_train_data(out_file_path):
    """
    WARNING: erases all the data in the file in 'out_file_path'
    """
    out_file = open(out_file_path, 'w').close()


def get_numerical_score(score):
    """
    (PovScore) --> +int if white leading, -int if black leading, 0 if even

    If mate() or mate_given(), return a signal to skip terminate the game
    """
    # since white is positive, will be negative if black is leading, so
    if score.is_mate():
        return None
    return float(score.white().score())



# 2. By using running move by move what stockfish would pick (no need for me to create my own tree stuff)
#    just use the one level "for move in move, take best move" code. Need to write it haha



SF_DEPTH = 10 # depth for stockfish

ANALYSIS_TIME = 0.1 # in seconds

path = "C:/Users/Ethan/Documents/GitHub/monty/lc0-v0.26.1-windows-gpu-nvidia-cuda/lc0.exe"

engine = chess.engine.SimpleEngine.popen_uci(path)
engine.options['Ponder'] = False

out_file_path = "C:/Users/Ethan/Documents/GitHub/monty/training_out_file"

train_path_tal = "C:/Users/Ethan/Documents/GitHub/monty/Tal.pgn"

train_path_carlsen = "C:/Users/Ethan/Documents/GitHub/monty/Carlsen.pgn"

new_data_out_file_path = "C:/Users/Ethan/Documents/GitHub/monty/new_data_out_file.txt"
test_file_out_path = "C:/Users/Ethan/Documents/GitHub/monty/test_file_out_path.txt"


if __name__ == "__main__":
    #erase_train_data(out_file_path) # erasing isn't the end of the world since i have the first 1000 tal games saved, but still avoid.
    tal_file = open(train_path_tal)
    arb_lim = 10000
    generate_train_data_from_PGN(tal_file, "C:/Users/Ethan/Documents/GitHub/monty/test_file_out_path.txt", limit=arb_lim)
    print("generated")
