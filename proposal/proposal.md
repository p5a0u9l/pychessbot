## EE596 Final Project Proposal
###### A Python Bot to Play Chess Using Computer Vision

#### Description
Build a classifier to classify images of chess pieces and use the model to predict
pieces and determine the position of a chess board. Given the position, use a
Python chess engine to get the best response move. Use an OS Api to make the move
and wait for a new position. The goal is to play in real time against real people
on Chess.com.

#### Algorithm Outline
##### Train a Bot
* Collect training images of Chess pieces
    - Screenshot of a fixed size chess board
    - Break into grid
    - Manually label pieces
* Train a model to recognize a chess piece

##### Play Chess
* Set up
    - Bring up Chess.com
    - Start Bot
* Make a Move
    - Take a screenshot of the chess board
    - Extract the board region and grid into squares
    - Classify pieces by square
    - Put together current position by square location and piece
    - Feed position to chess engine and get best move
    - Translate move into screen coordinates
    - Have Python click target piece and move to target position
    - Monitor clock to indicate next move and repeat