import Board
class Game:

    def play_move_on_np_board(board, move):
        move_str = str(move)
        if board.white_can_castle("A", move) and (move_str == 'e1g1' or move_str=='e1c1') and board.get_turn() == Board.WHITES_TURN:
            # Then king is castling. Error handling:
            if move_str == 'e1g1': # yes, I know i check twice. Live with it.
                board.castle_king_side('w', move)
            else:
                board.castle_queen_side('w', move)
        elif board.black_can_castle("A") and (move_str == 'e8g8' or move_str=='e8c8') and board.get_turn() == Board.BLACKS_TURN:
                if move_str == 'e8g8':
                    board.castle_king_side('b',move)
                else:
                    board.castle_queen_side('b', move)   
        else:
            board.play_move(move)