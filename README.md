# Monty - In Memoriam: 1 Decemeber 2007 - 1 July 2020. Best Dog.

## An ML chess AI using Keras

A Keras and Tensorflow Convolutional Neural Netowork (CNN) will be used to learn Stockfish's evaluation of positions.  
Then, all legal moves will be analyzed, and the move with the best score for the player moving will be played.

## Configuration

* Tensorflow - Version 2.1.0
* CUDA - Version 10.1
* CUDN - Version 7.6.5

This configuration is picky, and must be built following the appropriate Tensorflow and NVIDIA guidelines.

## TODO:

Since Keras only accepts `Tensors` of fixed size, a `Tensor` representing a chess position will be  
a one-hotted.

I've experimented a lot with different representations of data without training. My goal is to generate good data that will
converge quickly.

Here are my findings so far:

* Sparse data with lots of 0's does not perform well. 
* Ambiguous data (i.e. encoding how many times a square is attacked, but not which pieces attack it) does not perform well
* Adding linearly separable data that affects Stockfish's evaluation (material advantage, number of pieces on the board) improves convergence
* Testing with smaller data sets to see an initial convergence rate helps you judge the strength of a data representation
    

## Data Generation Life Cycle:  
  
I plan on using PGN to parse games and generate evaluation data  
using Stockfish. The game itself will be played in Python Chess.

1. Download a Portable Game Notation (PGN) file from an external source.  
2. For each game in the file:
3. Convert each position to the corresponding Tensor
4. Compute the position's score according to an established chess engine (Stockfish in this case)
5. Store the position in a tuple of (tensor, Stockfish score) in an output file

## Training:

Then simply train on the data stored in the output file.
