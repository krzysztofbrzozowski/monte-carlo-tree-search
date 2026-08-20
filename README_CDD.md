0. state = np.zeros((3, 3))

1. board_state = TicTacToeGameState(state = state, next_to_move=1)
-> no move has been done yet
-> next_to_move only rells Xes will move next

2. root = TwoPlayerMCTSNode(state=board_state)
-> pass the board_state to the TwoPlayerMCTSNode
-> root can be treated as new root of the tree

3. mcts = MonteCarloTreeSearch(root)
-> pass the root node to the **searching** wrapper MonteCarloTreeSearch

best_node = mcts.best_action(simulations_number=10)
-> start looking for the best action to take

Now inside 3.
-> expand current state
    -> get the legal actions for current move
    -> give a "virtual" move (on a copy of the object)
-> in this object play the game until the termination and write the result to reward
-> backpropagate the results from bottom to the top
-> find the best node and make this move

# Notes
## Note for the game coordinates
```python
#       TicTacToeMove coordinates map directly to NumPy board indices:
#       
#       x:0 y:0 -> board[0, 0] -> row 0, column 0
#       x:0 y:1 -> board[0, 1] -> row 0, column 1
#       x:1 y:0 -> board[1, 0] -> row 1, column 0
#       x:2 y:2 -> board[2, 2] -> row 2, column 2
#       
#       For a 3x3 board:
#       
#                 columns
#                  0  1  2
#               +---------
#       row 0   | 00 01 02
#       row 1   | 10 11 12
#       row 2   | 20 21 22
#       
#       In this code:
#       x_coordinate = row
#       y_coordinate = column
```python