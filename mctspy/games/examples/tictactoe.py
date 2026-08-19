import numpy as np
from mctspy.games.common import TwoPlayersAbstractGameState, AbstractGameAction


class TicTacToeMove(AbstractGameAction):
    def __init__(self, x_coordinate, y_coordinate, value):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.value = value

    def __repr__(self):
        return "x:{0} y:{1} v:{2}".format(
            self.x_coordinate,
            self.y_coordinate,
            self.value
        )


class TicTacToeGameState(TwoPlayersAbstractGameState):

    x = 1
    o = -1

    def __init__(self, state, next_to_move=1, win=None):
        if len(state.shape) != 2 or state.shape[0] != state.shape[1]:
            raise ValueError("Only 2D square boards allowed")
        self.board = state
        self.board_size = state.shape[0]
        if win is None:
            win = self.board_size
        self.win = win
        self.next_to_move = next_to_move

    @property
    def game_result(self):
        # Check horizontal and vertical wins
        for i in range(self.board_size - self.win + 1):

            # Take self.win consecutive rows and sum vertically
            #
            # Example (5x5):
            # 0  1  0  0 -1
            # 0  1 -1  0  0
            # 0  1  0 -1  0
            # ↓  ↓  ↓  ↓  ↓
            # 0  3 -1 -1 -1
            rowsum = np.sum(self.board[i:i + self.win], axis=0)

            # Take self.win consecutive columns and sum horizontally
            #
            # Example:
            # 1  0  0  → 1
            # 0  0  1  → 1
            # 1  1  1  → 3
            # 0  0  0  → 0
            # -1 0  0  → -1
            colsum = np.sum(self.board[:, i:i + self.win], axis=1)

            if rowsum.max() == self.win or colsum.max() == self.win:
                return self.x

            if rowsum.min() == -self.win or colsum.min() == -self.win:
                return self.o

        # Check every possible self.win x self.win sub-board
        for i in range(self.board_size - self.win + 1):
            for j in range(self.board_size - self.win + 1):

                sub = self.board[
                    i:i + self.win,
                    j:j + self.win
                ]

                # Main diagonal, top-left -> bottom-right
                #s
                # X . .
                # . X .
                # . . X
                diag_sum_tl = sub.trace()

                # Reverse rows and trace to check the opposite diagonal
                #
                # . . X
                # . X .
                # X . .
                diag_sum_tr = sub[::-1].trace()

                if diag_sum_tl == self.win or diag_sum_tr == self.win:
                    return self.x

                if diag_sum_tl == -self.win or diag_sum_tr == -self.win:
                    return self.o

        # Board is full and nobody won
        # If there is at least one 0 -> False -> do not enter the if
        # If there are no 0s         -> True  -> enter the if
        if np.all(self.board != 0):
            return 0.0

        # Game is not over
        return None

    def is_game_over(self):
        return self.game_result is not None

    def is_move_legal(self, move):
        # check if correct player moves
        if move.value != self.next_to_move:
            return False

        # check if inside the board on x-axis
        x_in_range = (0 <= move.x_coordinate < self.board_size)
        if not x_in_range:
            return False

        # check if inside the board on y-axis
        y_in_range = (0 <= move.y_coordinate < self.board_size)
        if not y_in_range:
            return False

        # finally check if board field not occupied ye
        return self.board[move.x_coordinate, move.y_coordinate] == 0

    
    def move(self, move):
        # Verify if move is legal
        if not self.is_move_legal(move):
            raise ValueError(
                "move {0} on board {1} is not legal". format(move, self.board)
            )
        # Copy the board to local variable
        new_board = np.copy(self.board)
        # Assign to the board the value (v) and the coordinates (x, y)
        new_board[move.x_coordinate, move.y_coordinate] = move.value
        # Give next move to opponent
        if self.next_to_move == self.x:
            next_to_move = self.o
        else:
            next_to_move = self.x
        # Create and return a new independent game-state object
        # The current state stays unchanged because new_board is a copy of self.board
        return type(self)(new_board, next_to_move, self.win)

    def get_legal_actions(self):
        indices = np.where(self.board == 0)
        return [
            TicTacToeMove(coords[0], coords[1], self.next_to_move)
            for coords in list(zip(indices[0], indices[1]))
        ]
