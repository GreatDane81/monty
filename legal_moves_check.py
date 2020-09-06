import chess
from chess.engine import Cp, Mate, MateGiven
import chess.pgn

board = chess.Board()
e_push = chess.Move.from_uci("e2e4")
board.push(e_push)
n = chess.Move.from_uci("b8a6")
board.push(n)
d_push = chess.Move.from_uci('d2d4')
board.push(d_push)
print(board.legal_moves)