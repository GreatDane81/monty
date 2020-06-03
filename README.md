# Monty

## An ML chess AI using Keras

ML will be used to learn Stockfish's evaluation of positions.  
This will be used alongside the "Min/Max" chess decision algorithm to make moves.

## TODO:

Since Keras only accepts `Tensors` of fixed size, a `Tensor` representing a chess position will be  
a one-hotted matrix of size:    
  
* 9x8x12  
	* Where 8x8 is used to to represent each square on the board  
	* One row, will be used to represent whose turn it is, will be filled 0's except for the leading bit  (0 for white's turn, 1 for black's)
	* Each of the 8x8 squares will have 1 if they have contain a piece. If they contain a piece, they will be one-hotted on the piece which   
	  occupies the square (white/black: king, queen, rook, bishop, knight, pawn)