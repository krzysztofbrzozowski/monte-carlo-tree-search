import time
class MonteCarloTreeSearch(object):

    def __init__(self, node):
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        if simulations_number is None :
            assert(total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else :
            for _ in range(0, simulations_number):
                # Initial state:
                #   Expand the current tree -> return child node 
                #     with next_state <- independent game-state (board) object
                #   Assign it to v for now         
                v = self._tree_policy()
                # Initial state:
                #   For that child node play game until termination happen 
                #   (some game result)
                #   reward <- game_result
                reward = v.rollout()
                # TODO -> analyze next
                v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        # This is current root <- TwoPlayerMCTSNode(state=board_state)
        # Initial state:
        #   children = []
        #   n = 0.0 -> because of @property
        current_node = self.root
        # While not game_over() -> self.game_result is not None (False)
        while not current_node.is_terminal_node():
            # Initial state:
            #   len(self.untried_actions) == 0 -> at the beginning 9 -> 9 == 0 ? -> False
            # untried_actions -> get_legal_actions (in this case)
            #
            #   See the Notes -> Note for the game coordinates to see game coordinates
            #
            # not False? -> go to current_node.expand()
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
