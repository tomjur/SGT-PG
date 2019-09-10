from ompl import base as ob
from ompl import geometric as og


class Planner:
    def __init__(self, point_robot_manager):
        self.point_robot_manager = point_robot_manager
        # create an SE2 state space
        space = ob.SE2StateSpace()
        # set lower and upper bounds
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, -point_robot_manager.dimension_length)
        bounds.setLow(1, -point_robot_manager.dimension_length)
        bounds.setHigh(0, point_robot_manager.dimension_length)
        bounds.setHigh(1, point_robot_manager.dimension_length)
        space.setBounds(bounds)
        self.space = space

    def is_state_valid(self, state):
        position = (state.getX(), state.getY())
        return self.point_robot_manager.is_free(position)

    def _get_state(self, coordinates):
        s = ob.State(self.space)
        s.get().setX(coordinates[0])
        s.get().setY(coordinates[1])
        return s

    # def get_free_random_point(self):
    #     s = ob.State(self.space)
    #     s.random()
    #     while not self.is_state_valid(s.get()):
    #         s.random()
    #     return s

    # def get_start_goal_pair(self):
    #     while True:
    #         start = self.get_free_random_point()
    #         goal = self.get_free_random_point()
    #         s1 = start.get()
    #         s2 = goal.get()
    #         if self.environment.line_intersects_main_obstacle((s1.getX(), s1.getY()), (s2.getX(), s2.getY())):
    #             return start, goal

    def plan(self, start, goal, plan_time=20.0, simplify=True):
        # create a simple setup object
        ss = og.SimpleSetup(self.space)
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        ss.setStartAndGoalStates(self._get_state(start), self._get_state(goal))
        # this will automatically choose a default planner with
        # default parameters
        solved = ss.solve(plan_time)
        if solved:
            # try to shorten the path
            if simplify:
                ss.simplifySolution()
            path = ss.getSolutionPath()
            return [(s.getX(), s.getY()) for s in path.getStates()]
        return None


if __name__ == '__main__':
    from point_robot_manager import PointRobotManager

    # obstacle_defs = [
    #     # obstacle1
    #     4, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5,
    #     # obstacle2
    #     4, -0.5, 0.65, 0.5, 0.65, 0.5, 0.55, -0.5, 0.55,
    # ]
    # point_robot_manager = PointRobotManager(obstacle_defs)
    # path1 = [(-0.75, -0.75), (-0.75, 0.75), (0.75, 0.75)]
    #
    # p = Planner(point_robot_manager)
    # path2 = p.plan(start=(-0.75, -0.75), goal=(0.75, 0.75))
    # print path2
    # f = PointRobotManager(obstacle_defs).plot(main_path=path1, support_path=path2)
    # f.savefig('/data/temp/fig1.png')

    # obstacle_defs = [
    #     # obstacle1
    #     4, -0.8, 1.0, -0.6, 1.0, -0.6, -0.8, -0.8, -0.8,
    #     # obstacle2
    #     4, -0.4, 0.8, -0.2, 0.8, -0.2, -1.0, -0.4, -1.0,
    #     # obstacle3
    #     4, 0.0, 1.0, 0.2, 1.0, 0.2, -0.8, 0.0, -0.8,
    #     # obstacle4
    #     4, 0.4, 0.8, 0.6, 0.8, 0.6, -1.0, 0.4, -1.0,
    #     # obstacle3
    #     4, 0.8, 1.0, 1.0, 1.0, 1.0, -0.8, 0.8, -0.8,
    # ]
    # point_robot_manager = PointRobotManager(obstacle_defs)
    # path1 = [
    #     (-0.9, 0.9), (-0.9, -0.9), (-0.5, -0.9), (-0.5, 0.9), (-0.1, 0.9), (-0.1, -0.9), (0.3, -0.9), (0.3, 0.9),
    #     (0.7, 0.9), (0.7, -0.9), (0.9, -0.9),
    # ]
    #
    # p = Planner(point_robot_manager)
    # path2 = p.plan(start=path1[0], goal=path1[-1])
    # print path2
    # f = PointRobotManager(obstacle_defs).plot(main_path=path1, support_path=path2)
    # f.savefig('/data/temp/fig1.png')

    # width = 0.01
    # obstacle_defs = [
    #     # obstacle1
    #     4, -0.5, 1.0, 0.5, 1.0, 0.5, width/2.0, -0.5, width/2.0,
    #     # obstacle2
    #     4, -0.5, -width/2.0, 0.5, -width/2.0, 0.5, -1.0, -0.5, -1.0
    # ]
    # point_robot_manager = PointRobotManager(obstacle_defs)
    # path1 = [
    #     (-0.75, 0.9), (-0.75, 0.0), (0.75, 0.0), (0.75, 0.9),
    # ]
    #
    # p = Planner(point_robot_manager)
    # # path2 = p.plan(start=path1[0], goal=path1[-1], simplify=False)
    # path2 = p.plan(start=path1[0], goal=path1[-1])
    # print path2
    # f = PointRobotManager(obstacle_defs).plot(main_path=path1, support_path=path2)
    # f.savefig('/data/temp/fig1.png')

    # width = 0.01
    # obstacle_defs = [
    #     # obstacle1
    #     4, -0.8, 1.0, 0.8, 1.0, 0.8, 0.5 + width, -0.8, 0.5 + width,
    #     # obstacle2
    #     4, 0.8, -1.0, -0.8, -1.0, -0.8, -0.5 - width, 0.8, -0.5 - width,
    #     # obstacle3
    #     4, -0.8, 0.5 + width, -0.5 - width, 0.5 + width, -0.5 - width, -0.5, -0.8, -0.5,
    #     # obstacle4
    #     4, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5 - width, -0.5, -0.5 - width,
    #     # obstacle5
    #     4, 0.5 + width, 0.5 + width, 0.8, 0.5 + width, 0.8, -0.5, 0.5 + width, -0.5
    # ]
    # point_robot_manager = PointRobotManager(obstacle_defs)
    # p = Planner(point_robot_manager)
    # path = p.plan(start=(-0.9, 0.9), goal=(0.9, -0.9))
    # print path
    # f = PointRobotManager(obstacle_defs).plot(main_path=path)
    # f.savefig('/data/temp/fig1.png')

    width = 0.01
    obstacle_defs = [
        # obstacle1
        4, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
        # obstacle2
        8, -0.8, 1.0, 0.8, 1.0, 0.8, width / 2., 0.5 + width, width / 2., 0.5 + width, 0.5 + width,
        -0.5 - width, 0.5 + width, -0.5 - width, width / 2., -0.8, width / 2.,
        # obstacle3
        8, -0.8, -1.0, -0.8, -width / 2., -0.5 - width, -width / 2., -0.5 - width, -0.5 - width,
        0.5 + width, -0.5 - width, 0.5 + width, -width / 2., 0.8, -width / 2., 0.8, -1.0,
    ]
    point_robot_manager = PointRobotManager(obstacle_defs)
    f = PointRobotManager(obstacle_defs).plot()
    f.savefig('/data/temp/fig1.png')
    # p = Planner(point_robot_manager)
    # path = p.plan(start=(-0.9, 0.9), goal=(0.9, -0.9))
    # print path
    # f = PointRobotManager(obstacle_defs).plot(main_path=path)
    # f.savefig('/data/temp/fig1.png')


