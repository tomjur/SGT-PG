import os
import numpy as np

from path_helper import get_base_directory


class TrajectoryNode:
    # state: None - unresolved, 0 - split, 1 - free, 2 - segment collision, 3 - endpoint collision
    def __init__(self, segment, budget):
        self.segment = segment
        self.budget = budget

        self.state = None

        self.middle_state = None
        self.right_subtree = None
        self.left_subtree = None

    def write_to_file(self, filepath):
        with open(filepath, 'w') as f:
            scan_order = [self]
            while len(scan_order) > 0:
                current = scan_order.pop(0)
                f.write('{}{}'.format(current.node_data_to_str(), os.linesep))
                if current.budget > 0:
                    scan_order.append(current.left_subtree)
                    scan_order.append(current.right_subtree)

    def node_data_to_str(self):
        b = self.budget
        s = self.segment[0].tolist()
        g = self.segment[1].tolist()
        m = self.middle_state
        if m is not None:
            m = m.tolist()
        state = self.state
        return '{}, {}, {}, {}, {}'.format(b, s, g, m, state)

    def _is_leaf(self):
        return self.left_subtree is None and self.right_subtree is None

    def print_status(self):
        if self._is_leaf():
            if self.state is None:
                state_desc = 'unresolved'
            elif self.state == 0:
                assert False
            elif self.state == 1:
                state_desc = 'free'
            elif self.state == 2:
                state_desc = 'segment collision'
            elif self.state == 3:
                state_desc = 'endpoint collision'
            else:
                assert False
            print('segment {},{} at budget: {} state is: {}'.format(
                self.segment[0], self.segment[1], self.budget, state_desc))
        else:
            self.left_subtree.print_status()
            self.right_subtree.print_status()

    def get_terminal_segments(self):
        if self._is_leaf():
            return [(self.segment, self.state)]
        else:
            return self.left_subtree.get_terminal_segments() + self.right_subtree.get_terminal_segments()


class AbstractMotionPlanningGame:
    def __init__(self, config):
        self.config = config
        lower, upper = self._get_state_bounds()
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.state_size = len(lower)

    @staticmethod
    def get_params_from_config(config):
        return os.path.join(get_base_directory(), 'scenario_params', config['general']['scenario'], 'params.pkl')

    def get_valid_states(self, states):
        return np.maximum(self.lower, np.minimum(self.upper, states))

    def get_random_state(self):
        return np.random.uniform(self.lower, self.upper, self.state_size)

    def get_free_random_state(self):
        while True:
            state = self.get_random_state()
            if self.is_free_state(state):
                return state

    def check_terminal_segments(self, segments):
        assert False

    def is_free_state(self, state):
        assert False

    def _get_state_bounds(self):
        return (0, ), (0, )

    def get_fixed_start_goal_pairs(self):
        assert False


class RLInterface:
    def __init__(self, config, game):
        self.config = config
        self.game = game

        self.start_joints = None
        self.goal_joints = None
        self.queries = None
        self.trajectory_tree = None
        self._query_id_to_node = None
        self._next_query_id = None
        self.collision_found = None

    def start_specific(self, traj, tree_depth):
        self.start_joints = np.array(traj[0])
        self.goal_joints = np.array(traj[-1])
        self.queries = {}
        self._query_id_to_node = {}
        self._next_query_id = 0
        self.collision_found = False

        segment = (self.start_joints, self.goal_joints)
        self.trajectory_tree = TrajectoryNode(segment, tree_depth)

        self._put_query(self.trajectory_tree)

    def _put_query(self, tree_node):
        if tree_node.budget > 0:
            query_id = self._next_query_id
            self._next_query_id += 1
            self.queries[query_id] = (tree_node.segment, tree_node.budget)
            self._query_id_to_node[query_id] = tree_node
        else:
            self._check_terminal_tree_node(tree_node)

    def _check_terminal_tree_node(self, tree_node):
        assert tree_node.state is None
        if self.game.check_terminal_segment(tree_node.segment):
            # free
            tree_node.state = 1
        else:
            # collision
            tree_node.state = 2
            self.collision_found = True

    def post_query_answer(self, query_id, middle_state):
        middle_state = self.game.get_valid_states(middle_state)
        # remove the query and get the query parameters
        segment, budget = self.queries.pop(query_id)
        current_tree_node = self._query_id_to_node[query_id]

        #  mark as split
        current_tree_node.state = 0
        # set middle
        current_tree_node.middle_state = middle_state
        # set the child nodes
        left_segment = (segment[0], middle_state)
        left_node = TrajectoryNode(left_segment, budget-1)
        current_tree_node.left_subtree = left_node
        right_segment = (middle_state, segment[1])
        right_node = TrajectoryNode(right_segment, budget-1)
        current_tree_node.right_subtree = right_node

        if budget > 1:
            # query the child nodes as well
            self._put_query(left_node)
            self._put_query(right_node)
        else:
            self._check_terminal_tree_node(left_node)
            self._check_terminal_tree_node(right_node)
