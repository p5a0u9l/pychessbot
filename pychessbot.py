#!/usr/bin/env python
import numpy as np
from time import sleep
import sys
bot_color = [False, True][int(sys.argv[1])]
from decoder import is_my_turn
from decoder import ChessBoardDecoder, ChessEngineIface, make_mouse_move


def make_a_move(cbd, fish):
    sleep_time = 1.0 + abs(2*np.random.randn())
    print "\nMy turn!... Snoozing %.1f sec..." % (sleep_time)
    sleep(sleep_time)
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
        print "Testing failed..."
        return None
    else:
        print "Testing passed, game on!"
        return cbd


def main():
    cbd = initialize_decoder()
    fish = ChessEngineIface()
    while 1:
        if cbd.last_board is None or is_my_turn(cbd):
            make_a_move(cbd, fish)
        else:
            sleep(1.0)


if __name__ == '__main__':
    main()
