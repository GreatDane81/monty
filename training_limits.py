# This file stores the number of games in each .pgn file.
# need this because np appending is O(n) for every use, so better to just use fixed size once
SIZES = {'Tal':2431,
          'Morphy':211
        }

# These scores represent the probability of white winning.
WHITE_MATE_SCORE = 1
BLACK_MATE_SCORE = 0