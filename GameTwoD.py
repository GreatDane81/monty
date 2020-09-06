class GameTwoD:

    def play_move_on_np_board(board, move):
        move_str = str(move)
        if board.white_can_castle("A") and (move_str == 'e1g1' or move_str=='e1c1'):
            # Then king is castling. Error handling:
            if move_str == 'e1g1': # yes, I know i check twice. Live with it.
                board.castle('W', 'K')
            else:
                board.castle('W','Q')
        elif board.black_can_castle("A") and (move_str == 'e8g8' or move_str=='e8c8'):
                if move_str == 'e8g8':
                    board.castle('B', 'K')
                else:
                    board.castle('B', 'Q') 
        else:
            board.push(move, move.promotion)