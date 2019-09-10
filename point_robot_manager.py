import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely import speedups
from descartes.patch import PolygonPatch


class PointRobotManager:
    def __init__(self, obstacles_definitions_list):
        self.use_speedups()
        # generate workspace bounding box
        self.dimension_length = 1.0
        # generate obstacles
        self.obstacles = self.parse_obstacles_definitions_list(obstacles_definitions_list)

    @staticmethod
    def use_speedups():
        if speedups.available:
            speedups.enable()

    @staticmethod
    def parse_obstacles_definitions_list(obstacles_definitions_list):
        obstacles = []
        current_index = 0
        while current_index < len(obstacles_definitions_list):
            number_of_obstacle_vertices = obstacles_definitions_list[current_index]
            current_index += 1
            obstacle_vertices = [
                [obstacles_definitions_list[current_index + 2*i], obstacles_definitions_list[current_index + 2*i+1]]
                for i in range(number_of_obstacle_vertices)
            ]
            obstacle = Polygon(obstacle_vertices, [])
            obstacles.append(obstacle)
            current_index += 2*number_of_obstacle_vertices
        return obstacles

    def is_free(self, coordinates):
        # check if in bounds
        if any(np.abs(coordinates) >= self.dimension_length):
            return False
        # check collision of robot with obstacles
        robot = Point(coordinates)
        if any([robot.intersection(obstacle) for obstacle in self.obstacles]):
            return False
        return True

    def is_linear_path_valid(self, state1, state2):
        # verify that both endpoints are valid
        if not self.is_free(state1) or not self.is_free(state2):
            return False
        # check the line segment in between
        path = LineString([state1, state2])
        if any([self._line_touch(path, polygon) for polygon in self.obstacles]):
            return False
        return True

    @staticmethod
    def _line_touch(line, obstacle):
        return line.intersects(obstacle)
        # return line.crosses(obstacle) or line.touches(obstacle)

    @staticmethod
    def v_color(flag):
        return '#6699cc' if flag else '#ffcc33'

    @staticmethod
    def plot_coords(ax, ob):
        x, y = ob.xy
        ax.plot(x, y, 'o', color='#999999', zorder=1)

    @staticmethod
    def plot_bounds(ax, ob):
        x, y = zip(*list((p.x, p.y) for p in ob.boundary))
        ax.plot(x, y, 'o', color='#000000', zorder=1)

    @staticmethod
    def plot_line(ax, ob):
        x, y = ob.xy
        ax.plot(x, y, color=PointRobotManager.v_color(ob.is_simple),
                alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)

    def plot(self, paths=None):
        from matplotlib import pyplot
        # return None
        fig = pyplot.figure(1, dpi=90)
        ax = fig.add_subplot(111)

        # plot bounding box and every obstacle
        # for polygon in [self.box] + self.obstacles:
        for polygon in self.obstacles:
            patch = PolygonPatch(
                polygon, facecolor=self.v_color(polygon.is_simple), edgecolor=self.v_color(polygon.is_simple),
                alpha=1.0, zorder=2
            )
            ax.add_patch(patch)

        def plot_path(path, path_color):
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            # ax.plot(xs, ys, '.-', color=path_color, alpha=0.3)
            ax.plot(xs, ys, '-', color=path_color, alpha=0.7)

        if paths is not None:
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'tan', 'teal', 'pink']
            for i, p in enumerate(paths):
                plot_path(p, colors[i])

        # print according to bounding box
        ax.set_xlim(-self.dimension_length, self.dimension_length)
        ax.set_ylim(-self.dimension_length, self.dimension_length)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)
        return fig


if __name__ == '__main__':
    obstacle_defs = [
        # obstacle1
        4, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5,
        # obstacle2
        4, -0.5, 0.65, 0.5, 0.65, 0.5, 0.55, -0.5, 0.55,
    ]
    path1 = [(-0.75, -0.75), (-0.75, 0.75), (0.75, 0.75)]
    path2 = [(-0.75, -0.75), (0.75, -0.75), (0.75, 0.75)]
    f = PointRobotManager(obstacle_defs).plot([path1, path2])
    f.savefig('/data/temp/fig1.png')


