import numpy as np
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.examples.connect4 import Connect4GameState

# Define inital state
state = np.zeros((7, 7))
# Object of Connect4GameState
board_state = Connect4GameState(
    state=state, next_to_move=np.random.choice([-1, 1]), win=4)

# link pieces to icons
pieces = {0: " ", 1: "X", -1: "O"}

# print a single row of the board
def stringify(row):
    return " " + " | ".join(map(lambda x: pieces[int(x)], row)) + " "

# display the whole board
def display(board):
    board = board.copy().T[::-1]
    for row in board[:-1]:
        print(stringify(row))
        print("-"*(len(row)*4-1))
    print(stringify(board[-1]))
    print()

display(board_state.board)

# keep playing until game terminates
while board_state.game_result is None:
    # calculate best move
    # At very begining we are looking
    # root  <- obiekt TwoPlayersGameMonteCarloTreeSearchNode(actual game state)
    #  ├─ node_0
    #  ├─ node_1
    #  ├─ node_2
    #  └─ node_3
    root = TwoPlayersGameMonteCarloTreeSearchNode(state=board_state)
    # -> MonteCarloTreeSearch is a high-level controller around the
    # TwoPlayersGameMonteCarloTreeSearchNode tree
    # -> It stores the root node and coordinates selection, rollout,
    # backpropagation, and final move choice
    mcts = MonteCarloTreeSearch(root)
    best_node = mcts.best_action(total_simulation_seconds=1)

    # update board
    board_state = best_node.state
    # display board
    display(board_state.board)

# print result
print(pieces[board_state.game_result])

if __name__ == "__main__":
    pass
