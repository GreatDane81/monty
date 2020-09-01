class Game:

    def play_move_on_np_board(board, move):
        move_str = str(move)
        if board.white_can_castle() and (move_str == 'e1g1' or move_str=='e1c1'):
            # Then king is castling. Error handling:
            if move_str == 'e1g1': # yes, I know i check twice. Live with it.
                board.castle_king_side('w')
            else:
                board.castle_queen_side('w')
        elif board.black_can_castle() and (move_str == 'e8g8' or move_str=='e8c8'):
                if move_str == 'e8g8':
                    board.castle_king_side('b')
                else:
                    board.castle_queen_side('b')   
        else:
            board.play_move(move, move.promotion)