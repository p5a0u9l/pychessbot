#!/usr/bin/env python
import numpy as np
import sys
import time
from decoder import ChessBoardDecoder, ChessEngineIface, make_mouse_move
bot_color = [False, True][int(sys.argv[1])]


def make_one_move(cbd, fish):
    sleep_time = 1.0 + abs(2*np.random.randn())
    print "\nMy turn!... Snoozing %.1f sec..." % (sleep_time)
    time.sleep(sleep_time)
    print "Decoding Current Board..."
    cbd.decode_board()
    cbd.board.turn = bot_color
    move = fish.get_best_move(cbd.board)[0]
    make_mouse_move(move.from_square, move.to_square, orient=bot_color)
    cbd.decode_board()  # force update of last board
    cbd.turn = ~bot_color   # ensure not bot's turn


def initialize_decoder():
    cbd = ChessBoardDecoder(bot_color)
    x, y = cbd.preprocess(retrain=False)
    cbd.train(x, y)
    if cbd.score_against_test_images < 1.0:
        print "Testing failed..."; return None
    else:
        print "Testing passed, game on!"; return cbd


def is_my_turn(cbd):
    cbd.decode_board()
    cbd.whose_turn()
    return cbd.turn == bot_color


def main(cbd):
    fish = ChessEngineIface()
    while 1:
        if cbd.last_board is None or is_my_turn(cbd):
            make_one_move(cbd, fish)
        else:
            time.sleep(1)


if __name__ == '__main__':
    main(initialize_decoder())
