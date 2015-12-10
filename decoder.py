__file__ = "decoder.py"
__author__ = "Paul Adams"

from sklearn import svm
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, ImageCollection, imsave
from pyscreenshot import grab
import pyautogui
import chess
from chess import uci
import numpy as np
import os
import sys
from os.path import join
# global board dimensions
# 1036, 163, 304, 30
x = 1036
y = 163
w = 304
h = 304
sq = w/8

board_lu = {}
board_lu[True] = np.array(
    [[(x + sq * r + sq / 2, y + w - sq * c - sq / 2) for r in range(8)]
     for c in range(8)]).ravel().reshape((64, 2))
board_lu[False] = np.array(
    [[(x + w - sq * r - sq / 2, y + sq * c + sq / 2) for r in range(8)]
     for c in range(8)]).ravel().reshape((64, 2))


class ChessBoardDecoder():

    """docstring for ChessBoardDecoder"""

    def __init__(self, color):
        self.clf = svm.SVC(gamma=0.001, kernel="linear", C=100)
        self.SCALE = 20
        self.label_file = "labeled_train_data.npy"
        self.n_dir = 10
        self.sym_lu = {0: "", 1: "p", 2: "n", 3: "b", 4: "r", 5: "q", 6: "k",
                       7: "P", 8: "N", 9: "B", 10: "R", 11: "Q", 12: "K"}
        self.color = color
        self.last_board = None
        self.turn = color
        self.board = chess.Board()
        self.print_flag = True

    def test(self, x, true_symbol):
        rect_piece = self.predict_piece(x)
        # print "True: %s, Predict: %s" % (true_symbol, rect_piece.symbol())
        return (true_symbol == rect_piece.symbol())*1

    def train(self, x, y):
        print "Fitting Features to Labels ..."
        self.clf.fit(x, y)

    def decode_board(self):
        im = grab_board()
        self.last_board = self.board
        board = chess.Board()
        for r in range(8):
            for c in range(8):
                grid_im = im[sq*r:sq*(r+1), sq*c:sq*(c+1), :]
                piece = self.predict_piece(self.im2feature(grid_im))
                if self.color:
                    board.set_piece_at((7 - r)*8 + c, piece)
                else:
                    board.set_piece_at(r*8 + 7 - c, piece)

        self.board = chess.Board(board.fen())

    def whose_turn(self):
        if self.board != self.last_board:
            self.turn = ~self.turn

        if self.print_flag:
            #        Turn   Turn
            # Color   0,0 | 0, 1   Me, Opp ==> xnor
            # Color   1,0 | 1, 1   Opp, Me
            print "\n%s (%s) to move ..." % \
                (["Black", "White"][self.turn],
                 ["Opponent", "Me"][not self.turn ^ self.color]),
            self.print_flag = False
            sys.stdout.flush()
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    def predict_piece(self, x):
        y = self.clf.predict(x)[0]
        sym = self.sym_lu[y]
        piece = chess.Piece.from_symbol(sym)
        return piece

    def preprocess(self, retrain=True):
        print "Labeling Features..."
        if retrain or not os.path.exists(self.label_file):
            data = []
            labels = []
            root = "train/"
            ic = ImageCollection(root + "*.png", load_func=self.im_loader)

            for j, im in enumerate(ic):
                code = [int(x) for x in ic.files[j][6:9]]
                piece_code = code[1] + 6*code[0]
                # print "Im: %s, Piece: %s" % (ic.files[j],
                # chess.PIECE_SYMBOLS[code[1]])
                labels.append(piece_code)
                data.append(np.array(im))

            np.save(self.label_file, [data, labels])
            X = np.zeros((len(data), self.SCALE**2))
            for i, a in enumerate(data):
                X[i, :] = np.array(a, np.float)
            y = np.array(labels, np.int)
        else:
            x = np.load(self.label_file)
            X = np.zeros((len(x[0]), self.SCALE**2))
            for i, a in enumerate(x[0]):
                X[i, :] = np.array(a, np.float)
            y = np.array(x[1], np.int)
        return X, y

    def im2feature(self, im):
        im = resize(im, (self.SCALE, self.SCALE))
        im = im.reshape((-1, 3))
        im = np.mean(im, axis=1)
        return im

    def im_loader(self, imname):
        im = imread(imname)[:, :, :3]
        return self.im2feature(im)

    def score_against_test_images(self):
        print "Scoring Classifier accuracy...",
        correct = []
        for i in range(self.n_dir):
            root = "train/"
            ic = ImageCollection(root + "*.png", load_func=self.im_loader)
            for j, im in enumerate(ic):
                code = [int(x) for x in ic.files[j][6:9]]
                piece = chess.Piece(code[1], [False, True][code[0]])
                c = self.test(im, piece.symbol())
                correct.append(c)
        score = np.sum(correct)/np.float(len(correct))
        print "Per Cent Correct = %.2f" % (score)
        return score


class ChessEngineIface():

    """docstring for ChessEngineIface"""

    def __init__(self):
        self.eng = uci.popen_engine("/usr/games/stockfish")
        self.eng.uci()

    def get_best_move(self, board):
        print board
        self.eng.ucinewgame()
        self.eng.isready()
        self.eng.position(board)
        print "Stockfish is pondering... ",
        move = self.eng.go(movetime=500)
        print "Best move is %s... " % (move[0].uci()),
        return move


def make_mouse_move(from_sq, to_sq, orient=True):
    print "Moving from square %d to square %d" % (from_sq, to_sq)
    # click on from square
    x = board_lu[orient][from_sq][0] + sq/4*np.random.random_sample()
    y = board_lu[orient][from_sq][1] + sq/4*np.random.random_sample()
    pyautogui.moveTo(x, y, duration=0.1 + np.random.random_sample())
    # click on to square
    x = board_lu[orient][to_sq][0] + sq/4*np.random.random_sample()
    y = board_lu[orient][to_sq][1] + sq/4*np.random.random_sample()
    pyautogui.dragTo(x, y, duration=0.1 + np.random.random_sample())


def grab_board(x=x, y=y, w=w, h=h):
    # print "Grabbing board image at (%d, %d) x (%d, %d) ..." % (x, y, x + w,
    # y + h),
    im = grab(bbox=(x, y, x + w, y + h), backend='scrot')
    im.save('screenshot.png')
    return plt.imread('screenshot.png')


def cache_squares(im, dest="pieces"):
    # Given board image, save squares to pieces directory in row/col order
    print "Caching squares from image to %s..." % (dest),
    sq = w/8
    if not os.path.exists(dest):
        os.mkdir(dest)
    for r in range(8):
        for c in range(8):
            piece = im[sq*r:sq*(r+1), sq*c:sq*(c+1), :]
            piecename = join(dest, str(r) + str(c) + ".png")
            imsave(piecename, piece)
    print "Success."


def play_stockfish(board):
    eng = uci.popen_engine("/usr/games/stockfish")
    eng.uci()
    eng.position(board)
    m = eng.go(movetime=1000)
    eng.ucinewgame()
    # import ipdb; ipdb.set_trace()
    make_mouse_move(m[0].from_square, m[0].to_square)
    board.push_uci(m[0].uci())
    print board


def is_my_turn(cbd, bot_color):
    cbd.decode_board()
    cbd.whose_turn()
    return cbd.turn == bot_color


def check_align(x, y, w, h):
    sq = w/8
    im = grab_board(x=x, y=y, w=w, h=h)
    plt.imshow(im)
    plt.show()
    plt.imshow(im[:sq, :sq, :])
    plt.show()
    plt.imshow(im[7*sq:, 7*sq:, :])
    plt.show()
