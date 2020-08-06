import chess
import chess.engine


path = "C:/Users/Ethan Dain/Desktop/University/Machine Learning/Code/monty/stockfish/stockfish-11-win/stockfish-11-win/Windows/stockfish_20011801_x64.exe"


engine = chess.engine.SimpleEngine.popen_uci(path)

board = chess.Board()
while not board.is_game_over():
    result = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(result.move)
    info = engine.analyse(board, chess.engine.Limit(time=0.1))
    print(board)
    print("Score:", info["score"])

engine.quit()
# this works 