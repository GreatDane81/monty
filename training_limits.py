# This file stores the number of games in each .pgn file.
# need this because np appending is O(n) for every use, so better to just use fixed size once
SIZES = {'Tal':2431,
          'Morphy':211
        }

# Of course a mate is "infinitely" better than a totally winning non-mate position,
# but the model will expect some value to train with so I'm setting a constant +/- 20k
# because analysis shows the top winning/losing scores are 15k.
WHITE_MATE_SCORE = 20000
BLACK_MATE_SCORE = -20000