#!/usr/bin/env python
# import chess
import numpy as np
import chess
from skimage.io import ImageCollection
from sklearn_decoder import ChessPieceRecognizer
from chess import uci
from create_test_train import make_move
import sys
import time
COLOR = int(sys.argv[1])
eng = uci.popen_engine("/usr/games/stockfish")
eng.uci()


def do_test(cpr):
    correct = []
    lu = {0: False, 1: True}
    for i in range(cpr.n_dir):
        root = "train/"
        ic = ImageCollection(root + "*.png", load_func=cpr.load_func)
        for j, im in enumerate(ic):
            code = [int(x) for x in ic.files[j][6:9]]
            piece = chess.Piece(code[1], [False, True][code[0]])
            c = cpr.test(im, piece.symbol())
            correct.append(c)
    score = np.sum(correct)/np.float(len(correct))
    print "PerCent Correct = %.2f" % (score)
    return score


def play_once(cpr):
    sleep_time = 1 + abs(10*np.random.randn())
    print "Sleeping for %d..." % (sleep_time)
    time.sleep(sleep_time)
    print "Decoding Current Board...",
    turn, board = cpr.decode_board()
    board.turn = [False, True][COLOR]
    move = get_stockfish_response(board)
    print "Best move is %s..." % (move[0].uci())
    make_move(move[0].from_square, move[0].to_square)


def get_stockfish_response(board):
    print board
    print "%s to move..." % (["Black", "White"][COLOR])
    eng.ucinewgame(); eng.isready()
    eng.position(board)
    print "Stockfish is pondering..."
    m = eng.go(movetime=500)
    return m


def main():
    cpr = ChessPieceRecognizer()
    print "Labeling Features..."
    x, y = cpr.label_features(retrain=False)
    print "Fitting Features to Labels ..."
    cpr.clf.fit(x, y)
    print "Testing Classifier...",
    if do_test(cpr) < 1.0:
        print "Testing failed...",
        return
    play_once(cpr)

if __name__ == '__main__':
    while 1:
        main()
        import ipdb; ipdb.set_trace()
