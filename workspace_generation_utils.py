import Image
import pickle
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate
# from matplotlib import pyplot
from descartes.patch import PolygonPatch


class WorkspaceParams:
    def __init__(self):
        self.number_of_obstacles = 0
        self.centers_position_x = []
        self.centers_position_z = []
        self.sides_x = []
        self.sides_z = []
        self.y_axis_rotation = []
        self.rays = []

    def save(self, file_path):
        pickle.dump(self, open(file_path, 'w'))

    @staticmethod
    def load_from_file(file_path):
        instance = pickle.load(open(file_path))
        shrink = 0.7
        # shrink = 1.0
        instance.sides_x = [s * shrink for s in instance.sides_x]
        instance.sides_z = [s * shrink for s in instance.sides_z]
        return instance
        # return pickle.load(open(file_path))

    @staticmethod
    def _get_box_polygon(center_x, center_z, side_x, side_z, y_rotation):
        points = [
            (center_x - side_x / 2.0, center_z + side_z / 2.0 - 0/125),
            (center_x + side_x / 2.0, center_z + side_z / 2.0 - 0/125),
            (center_x + side_x / 2.0, center_z - side_z / 2.0 - 0/125),
            (center_x - side_x / 2.0, center_z - side_z / 2.0 - 0/125),
        ]
        box = Polygon(points)
        box = rotate(geom=box, angle=-y_rotation, origin='center', use_radians=True)
        return box

    # def print_image(self, trajectory=None, reference_trajectory=None, starting_pose=None, trajectory_end_pose=None,
    #                 reference_end_pose=None):
    #     fig = pyplot.figure(1, dpi=90)
    #     ax = fig.add_subplot(111)
    #
    #     # plot obstacles
    #     for i in range(self.number_of_obstacles):
    #         rotated_box = WorkspaceParams._get_box_polygon(self.centers_position_x[i], self.centers_position_z[i],
    #                                                        self.sides_x[i], self.sides_z[i], self.y_axis_rotation[i])
    #         patch = PolygonPatch(rotated_box, facecolor='#6699cc', edgecolor='#6699cc', alpha=1.0, zorder=2)
    #         ax.add_patch(patch)
    #
    #     def plot_path(path, path_color):
    #         xs = [p[0] for p in path]
    #         ys = [p[1] for p in path]
    #         ax.plot(xs, ys, '.-', color=path_color)
    #
    #     if trajectory is not None:
    #         plot_path(trajectory, 'red')
    #
    #     if reference_trajectory is not None:
    #         plot_path(reference_trajectory, 'green')
    #
    #     if starting_pose is not None:
    #         plot_path(starting_pose, 'cyan')
    #
    #     if trajectory_end_pose is not None:
    #         plot_path(trajectory_end_pose, 'magenta')
    #
    #     if reference_end_pose is not None:
    #         plot_path(reference_end_pose, 'black')
    #
    #     # print according to bounding box
    #     # x_range = [-int(0.5), int(self.outerbox_length)]
    #     # y_range = [-int(self.outerbox_length), int(self.outerbox_length)]
    #     x_range = [-0.5, 0.5]
    #     y_range = [0, 0.5]
    #     ax.set_xlim(*x_range)
    #     ax.set_ylim(*y_range)
    #     ax.axes.get_xaxis().set_visible(False)
    #     ax.axes.get_yaxis().set_visible(False)
    #     ax.set_aspect(1)
    #     return fig
    #
    # def print_image_many_trajectories(self, ax, other_trajectories, reference_trajectory=None):
    #     # fig = pyplot.figure(1, dpi=90)
    #     # ax = fig.add_subplot(111)
    #
    #     # plot obstacles
    #     for i in range(self.number_of_obstacles):
    #         rotated_box = WorkspaceParams._get_box_polygon(self.centers_position_x[i], self.centers_position_z[i],
    #                                                        self.sides_x[i], self.sides_z[i], self.y_axis_rotation[i])
    #         patch = PolygonPatch(rotated_box, facecolor='#6699cc', edgecolor='#6699cc', alpha=1.0, zorder=2)
    #         ax.add_patch(patch)
    #
    #     def plot_path(path, path_color):
    #         xs = [p[0] for p in path]
    #         ys = [p[1] for p in path]
    #         ax.plot(xs, ys, '.-', color=path_color)
    #
    #     for trajectory in other_trajectories:
    #         plot_path(trajectory, 'red')
    #
    #     if reference_trajectory is not None:
    #         plot_path(reference_trajectory, 'green')
    #
    #     # print according to bounding box
    #     # x_range = [-int(0.5), int(self.outerbox_length)]
    #     # y_range = [-int(self.outerbox_length), int(self.outerbox_length)]
    #     x_range = [-0.5, 0.5]
    #     y_range = [0, 0.5]
    #     ax.set_xlim(*x_range)
    #     ax.set_ylim(*y_range)
    #     ax.axes.get_xaxis().set_visible(False)
    #     ax.axes.get_yaxis().set_visible(False)
    #     ax.set_aspect(1)
    #     # return fig
    #     return ax
    #
    # @staticmethod
    # def _figure_to_nparray(fig):
    #     fig.canvas.draw()
    #     w, h = fig.canvas.get_width_height()
    #     buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    #     buf.shape = (w, h, 4)
    #     buf = np.roll(buf, 3, axis=2)
    #     return buf
    #
    # @staticmethod
    # def _figure_to_image(fig):
    #     buf = WorkspaceParams._figure_to_nparray(fig)
    #     w, h, d = buf.shape
    #     return Image.frombytes("RGBA", (w, h), buf.tobytes())
    #
    # @staticmethod
    # def _remove_transparency(im, bg_colour=(255, 255, 255)):
    #     if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
    #
    #         # Need to convert to RGBA if LA format due to a bug in PIL
    #         alpha = im.convert('RGBA').split()[-1]
    #
    #         # Create a new background image of our matt color.
    #         # Must be RGBA because paste requires both images have the same format
    #
    #         bg = Image.new("RGBA", im.size, bg_colour + (255,))
    #         bg.paste(im, mask=alpha)
    #         return bg
    #
    #     else:
    #         return im
    #
    # def get_image_as_numpy(self):
    #     f = self.print_image()
    #     im = WorkspaceParams._figure_to_image(f)
    #     im = WorkspaceParams._remove_transparency(im).convert('L')
    #     width = im.width / 16
    #     height = im.height / 16
    #     im.thumbnail((width, height), Image.ANTIALIAS)
    #     res = np.asarray(im)
    #     # res = np.array(im.getdata()).reshape((im.size[0], im.size[1], 1))
    #     # res = res.reshape((im.size[0], im.size[1], 1))
    #     pyplot.clf()
    #     return res
