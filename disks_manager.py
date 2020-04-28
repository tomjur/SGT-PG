from shapely.geometry import Point, Polygon, LineString
from shapely import speedups
import numpy as np


class DisksManager:
    def __init__(self):
        self.disk_radius = 0.21
        self.use_speedups()
        # generate workspace bounding box
        self.dimension_length = 1.0
        # for now obstacles are static
        max_d = 100.
        self.obstacle = Polygon(
            shell=[(-max_d, max_d), (max_d, max_d), (max_d, -max_d), (-max_d, -max_d)],
            holes=[
                # # no obs
                # [(-1., 1.), (1., 1.), (1., -1.), (-1., -1.)]
                # # regular obs
                # [(-1., 1.), (1., 1.), (1., 0.2), (0.4, 0.2), (0.4, -0.2), (1., -0.2), (1., -1.), (-1., -1.),
                #  (-1., -0.2), (-0.4, -0.2), (-0.4, 0.2), (-1., 0.2)]
                # # tall obs
                # [(-1., 1.), (1., 1.), (1., 0.25), (0.4, 0.25), (0.4, -0.25), (1., -0.25), (1., -1.), (-1., -1.),
                #  (-1., -0.25), (-0.4, -0.25), (-0.4, 0.25), (-1., 0.25)]
                # short obs
                [(-1., 1.), (1., 1.), (1., 0.1), (0.4, 0.1), (0.4, -0.1), (1., -0.1), (1., -1.), (-1., -1.),
                 (-1., -0.1), (-0.4, -0.1), (-0.4, 0.1), (-1., 0.1)]
            ])

    @staticmethod
    def use_speedups():
        if speedups.available:
            speedups.enable()

    def is_motion_free(self, circle_start_to_end_pos):
        # gets the motions as buffered line segments
        per_object_motions = [LineString([np.array(start), np.array(end)]).buffer(self.disk_radius)
                              for start, end in circle_start_to_end_pos]

        # get the individual intersections with the walls
        intersection_with_obstacles = [m.intersection(self.obstacle)
                                       for m in per_object_motions if m.intersects(self.obstacle)]

        # gets the pairwise intersections
        # pairwise_intersections = self._get_time_dependant_pairwise_intersections(circle_start_to_end_pos)
        pairwise_intersections = self._get_complete_perwise_intersections(per_object_motions)

        # get the union of all intersection types
        intersections = intersection_with_obstacles + pairwise_intersections

        if len(intersections) == 0:
            # there are no intersections, we have 1. for free size, and 0. for collision
            return 1., 0., self._get_minimal_distances(per_object_motions)

        intersection_union = intersections[0]
        for i in range(1, len(intersections)):
            intersection_union = intersection_union.union(intersections[i])
        intersection_area = intersection_union.area

        motions_union = per_object_motions[0]
        for i in range(1, len(per_object_motions)):
            motions_union = motions_union.union(per_object_motions[i])
        union_area = motions_union.area

        collision_area = intersection_area / union_area
        collision_area = max(collision_area, 0.)
        collision_area = min(collision_area, 1.)
        free_area = 1. - collision_area
        return free_area, collision_area, self._get_minimal_distances(per_object_motions)

    def _get_minimal_distances(self, per_object_motions):
        min_distance = None
        for i in range(len(per_object_motions)):
            m1 = per_object_motions[i]
            for j in range(i+1, len(per_object_motions)):
                m2 = per_object_motions[j]
                current_distance = m1.distance(m2)
                if min_distance is None or min_distance > current_distance:
                    min_distance = current_distance
        return min_distance

    def _get_time_dependant_pairwise_intersections(self, circle_start_to_end_pos):
        intersections = []
        for i in range(len(circle_start_to_end_pos)):
            si, gi = circle_start_to_end_pos[i]
            for j in range(i+1, len(circle_start_to_end_pos)):
                sj, gj = circle_start_to_end_pos[j]
                current_intersection = self._get_time_dependant_pairwise_intersection(si, sj, gi, gj)
                if current_intersection is not None:
                    intersections.append(current_intersection)
        return intersections

    def _get_time_dependant_pairwise_intersection(self, s1, s2, g1, g2):
        s1, s2, g1, g2 = np.array(s1), np.array(s2), np.array(g1), np.array(g2)
        ls1 = LineString([np.array(s1), np.array(g1)]).buffer(self.disk_radius)
        ls2 = LineString([np.array(s2), np.array(g2)]).buffer(self.disk_radius)
        if not ls1.intersects(ls2):
            return None
        circles1 = self._get_circles_by_step(s1, g1)
        circles2 = self._get_circles_by_step(s2, g2)
        if len(circles1) < len(circles2):
            circles1 = circles1 + [circles1[-1]] * (len(circles2)-len(circles1))
        elif len(circles2) < len(circles1):
            circles2 = circles2 + [circles2[-1]] * (len(circles1) - len(circles2))
        assert len(circles1) == len(circles2)
        intersections = []
        for i in range(len(circles1)):
            c1 = circles1[i]
            c2 = circles2[i]
            if c1.intersects(c2):
                intersections.append(c1.intersection(c2))
        if len(intersections) == 0:
            return None
        union = intersections[0]
        for i in range(1, len(intersections)):
            union = union.union(intersections[i])
        return union

    def _get_circles_by_step(self, s1, s2, step_size=0.01):
        result = [Point(s1).buffer(self.disk_radius)]
        if np.all(s1 == s2):
            return result
        d = np.linalg.norm(s2-s1)
        steps = int(np.floor(d / step_size))
        for i in range(steps):
            p1 = s1 + step_size * (i+1) * (s2 - s1)
            result.append(Point(p1).buffer(self.disk_radius))
        result.append(Point(s2).buffer(self.disk_radius))
        return result

    def _get_complete_perwise_intersections(self, per_object_motions):
        return [
            per_object_motions[i].intersection(per_object_motions[j])
            for i in range(len(per_object_motions)) for j in range(i+1, len(per_object_motions))
            if per_object_motions[i].intersects(per_object_motions[j])
        ]

    def is_pos_in_bounds(self, pos):
        if pos[0] <= -1. + self.disk_radius or pos[0] >= 1. - self.disk_radius:
            return False
        if pos[1] <= -1. + self.disk_radius or pos[1] >= 1. - self.disk_radius:
            return False
        return True


if __name__ == '__main__':
    disk_manager = DisksManager()
    poses = [
        [0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1], [0.8, 0.], [-0.8, 0.], [0., 0.8], [0., -0.8], [0.75, 0.],
        [-0.75, 0.], [0., 0.75], [0., -0.75]
    ]
    for p in poses:
        print(f'pose: {p}')
        print(f'is in bounds: {disk_manager.is_pos_in_bounds(p)}')
    motions = [
        [[[-0.6, 0.6], [0.6, 0.6]], [[0.6, -0.6], [-0.6, -0.6]]],
        [[[-0.6, 0.6], [0.6, 0.6]], [[-0.6, -0.6], [-0.6, -0.6]]],
        [[[-0.6, 0.6], [0.6, 0.6]], [[-0.6, -0.6], [-0.6, 0.6]]],
        [[[-0.6, 0.6], [0.6, 0.6]], [[0.6, 0.6], [-0.6, 0.6]]],
        [[[-0.6, 0.6], [0.6, 0.6]], [[0.6, -0.75], [-0.6, -0.6]]],
        [[[-0.6, 0.6], [0.6, 0.6]], [[0.6, -0.6], [-0.6, -0.75]]],
        [[[-0.6, 0.6], [0.6, 0.6]], [[0.6, -0.6], [-0.6, -0.6]], [[0., 0.], [0., 0.]]],
    ]
    for j, m in enumerate(motions):
        print(f'motion {j}')
        for i, (s, g) in enumerate(m):
            print(f'object {i} from {s} to {g}')
        print(f'is free? free area, collision area {disk_manager.is_motion_free(m)}')