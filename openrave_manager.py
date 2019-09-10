import random
import numpy as np
import time
from openravepy import *
from potential_point import PotentialPoint
from workspace_generation_utils import WorkspaceParams
from docker_path_helper import *


class OpenraveManager(object):
    def __init__(self, segment_validity_step):
        # env_path = os.path.abspath(
        #     os.path.expanduser('~/ModelBasedDDPG/config/widowx_env.xml'))
        env_path = os.path.join(get_config_directory(), 'widowx_env.xml')
        self.env = Environment()
        self.env.StopSimulation()
        self.env.Load(env_path)  # load a simple scene
        self.robot = self.env.GetRobots()[0] # load the robot
        self.links_names = [l.GetName() for l in self.robot.GetLinks()]
        self.robot.SetActiveDOFs(range(1, 5)) # make the first joint invalid
        # set the color
        color = np.array([33, 213, 237])
        for link in self.robot.GetLinks():
            for geom in link.GetGeometries():
                geom.SetDiffuseColor(color)
        self.objects = []
        self.segment_validity_step = segment_validity_step
        # translate the potential to list of (unprocessed_point, link, coordinate)
        self.potential_points = PotentialPoint.parse_value([5, -0.02, 0.035])
        self.joint_safety = 0.0001
        self.loaded_params_path = None
        self.loaded_params = None

    def load_params(self, workspace_params, params_path):
        if self.loaded_params_path is not None and self.loaded_params_path == params_path:
            # already loaded
            return
        with self.env:
            for i in range(workspace_params.number_of_obstacles):
                body = RaveCreateKinBody(self.env, '')
                body.SetName('box{}'.format(i))
                body.InitFromBoxes(np.array([[0, 0, 0, workspace_params.sides_x[i], 0.01, workspace_params.sides_z[i]]]),
                                   True)
                self.env.Add(body, True)

                transformation_matrix = np.eye(4)
                translation = np.array([
                    workspace_params.centers_position_x[i], 0.0, workspace_params.centers_position_z[i]])

                theta = workspace_params.y_axis_rotation[i]
                rotation_matrix = np.array([
                    [np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]
                ])
                transformation_matrix[:3, -1] = translation
                transformation_matrix[:3, :3] = rotation_matrix
                body.SetTransform(transformation_matrix)
                self.objects.append(body)
        self.loaded_params_path = params_path
        self.loaded_params = workspace_params

    def remove_objects(self):
        with self.env:
            while len(self.objects):
                body = self.objects.pop()
                self.env.Remove(body)
        self.loaded_params_path = None
        self.loaded_params = None

    def set_params(self, params_path):
        loaded = self.loaded_params_path
        if loaded is None:
            self.load_params(WorkspaceParams.load_from_file(params_path), params_path)
        else:
            if loaded != params_path:
                self.remove_objects()
                self.load_params(WorkspaceParams.load_from_file(params_path), params_path)

    def get_number_of_joints(self):
        return self.robot.GetDOF()

    def get_joint_bounds(self):
        return self.robot.GetDOFLimits()

    def _real_to_virtual(self, real_joints):
        assert real_joints[0] == 0.0
        lower, upper = self.get_joint_bounds()
        nominator = np.array(real_joints) - lower
        denominator = upper - lower
        result = nominator / denominator
        return tuple(result.tolist()[1:])

    def _virtual_to_real(self, virtual_joints):
        virtual_joints = np.array([0.5] + list(virtual_joints))
        lower, upper = self.get_joint_bounds()
        a = upper - lower
        result = a * virtual_joints + lower
        return tuple(result)

    def get_random_joints(self, fixed_positions_dictionary=None, return_virtual=False):
        joint_bounds = self.get_joint_bounds()
        result = []
        for i in range(self.get_number_of_joints()):
            if fixed_positions_dictionary is not None and i in fixed_positions_dictionary:
                result.append(fixed_positions_dictionary[i])
            else:
                result.append(random.uniform(joint_bounds[0][i], joint_bounds[1][i]))
        result = self.truncate_joints(result)
        if return_virtual:
            result = self._real_to_virtual(result)
        return tuple(result)

    def truncate_joints(self, joints):

        bounds = self.get_joint_bounds()
        res = list(joints)
        for i, j in enumerate(joints):
            lower = bounds[0][i] + self.joint_safety
            res[i] = max(res[i], lower)
            upper = bounds[1][i] - self.joint_safety
            res[i] = min(res[i], upper)
        return tuple(res)

    def is_valid(self, joints, is_virtual=False):
        if is_virtual:
            joints = self._virtual_to_real(joints)
        self.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
        res = not self.robot.CheckSelfCollision()
        # if not res:
        #     print 'self collision found'
        if res and self.objects is not None:
            for item in self.objects:
                res = res and not self.env.CheckCollision(self.robot, item)
            # if not res:
            #     print 'obstacle collision found'
        return res

    def plan(self, start_joints, goal_joints, max_planner_iterations):
        with self.env:
            if not self.is_valid(start_joints) or not self.is_valid(goal_joints):
                return None
            self.robot.SetDOFValues(start_joints, [0, 1, 2, 3, 4])
            manipprob = interfaces.BaseManipulation(self.robot)  # create the interface for basic manipulation programs
            try:
                items_per_trajectory_step = 10
                active_joints = self.robot.GetActiveDOF()
                # call motion planner with goal joint angles
                traj = manipprob.MoveActiveJoints(goal=goal_joints[1:], execute=False, outputtrajobj=True, maxtries=1,
                                                  maxiter=max_planner_iterations)
                # found plan, if not an exception is thrown and caught below
                traj = list(traj.GetWaypoints(0, traj.GetNumWaypoints()))
                assert len(traj) % items_per_trajectory_step == 0
                # take only the joints values and add the 0 joint.
                traj = [[0.0] + traj[x:x + items_per_trajectory_step][:active_joints] for x in
                        xrange(0, len(traj), items_per_trajectory_step)]
                # assert validity
                if self.get_last_valid_in_trajectory(traj) != traj[-1]:
                    return None
                # plan found and validated!
                return traj
            except Exception, e:
                print str(e)
                return None

    def check_segment_validity(self, start_joints, end_joints, is_virtual=False):
        if is_virtual:
            start_joints = self._virtual_to_real(start_joints)
            end_joints = self._virtual_to_real(end_joints)
        steps = self.partition_segment(start_joints, end_joints)
        random.shuffle(steps)
        for step in steps:
            if not self.is_valid(step):
                return False
        return True

    def partition_segment(self, start_joints, end_joints):
        # partition the segment between start joints to end joints
        current = np.array(start_joints)
        next = np.array(end_joints)
        difference = next - current
        difference_norm = np.linalg.norm(difference)
        step_size = self.segment_validity_step
        if difference_norm < step_size:
            # if smaller than allowed step just append the next step
            return [tuple(end_joints)]
        else:
            scaled_step = (step_size / difference_norm) * difference
            steps = []
            for alpha in range(int(np.floor(difference_norm / step_size))):
                processed_step = current + (1 + alpha) * scaled_step
                steps.append(processed_step)
            # we probably have a leftover section, append it to res
            last_step_difference = np.linalg.norm(steps[-1] - next)
            if last_step_difference > 0.0:
                steps.append(next)
            # append to list of configuration points to test validity
            return [tuple(s) for s in steps]

    def get_last_valid_in_trajectory(self, trajectory):
        for i in range(len(trajectory)-1):
            if not self.check_segment_validity(trajectory[i], trajectory[i+1]):
                return trajectory[i]
        return trajectory[-1]

    def get_initialized_viewer(self):
        if self.env.GetViewer() is None:
            self.env.SetViewer('qtcoin')
        # set camera
        camera_transform = np.eye(4)
        theta = -np.pi / 2
        rotation_matrix = np.array([
            [1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]
        ])
        camera_transform[:3, :3] = rotation_matrix
        camera_transform[:3, 3] = np.array([0.0, -1.0, 0.25])
        time.sleep(1)
        viewer = self.env.GetViewer()
        viewer.SetCamera(camera_transform)
        return viewer

    def get_links_poses(self, joints):
        self.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
        poses = self.robot.GetLinkTransformations()
        result = {
            link_name: tuple(poses[i][[0, 2], -1])
            for i, link_name in enumerate(self.links_names) if link_name in self.links_names
        }
        return result

    def get_links_poses_array(self, joints):
        poses = self.get_links_poses(joints)
        return [poses[link_name] for link_name in self.links_names]

    def get_potential_points_poses(self, joints, post_process=True):
        self.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
        link_transform = self.robot.GetLinkTransformations()
        result = {p.tuple: np.matmul(link_transform[p.link], p.coordinate) for p in self.potential_points}
        if post_process:
            result = {k: (result[k][0], result[k][2]) for k in result}
        return result

    def get_target_pose(self, joints):
        # target is the last potential
        return self.get_potential_points_poses(joints)[self.potential_points[-1].tuple]

    @staticmethod
    def _post_process_jacobian(j, is_numeric=False):
        return j[[0, 2], 1 if is_numeric else 0:].transpose()

    def get_links_jacobians(self, joints, modeling_links=None):
        if modeling_links is None:
            modeling_links = self.links_names
        self.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
        poses = self.robot.GetLinkTransformations()
        return {
            link_name: self._post_process_jacobian(self.robot.CalculateActiveJacobian(i, poses[i][:3, 3]))
            for i, link_name in enumerate(self.links_names) if link_name in modeling_links
        }

    def get_potential_points_jacobians(self, joints):
        potential_points_poses = self.get_potential_points_poses(joints, post_process=False)
        self.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
        return {
            p.tuple: self._post_process_jacobian(
                self.robot.CalculateActiveJacobian(p.link, potential_points_poses[p.tuple])
            )
            for p in self.potential_points
        }
