import time
import os

import pybullet as p
import numpy as np

from docker_path_helper import get_base_directory


class PandaSceneManager:
    # for reference see
    # https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

    def __init__(self, use_ui=True, robot_position=(0., 0., 0.3), position_sensitivity=0.0001):
        self.use_ui = use_ui
        self.position_sensitivity = position_sensitivity
        self.robot_base = robot_position
        # setup pybullet
        self.my_id = p.connect(p.GUI if use_ui else p.DIRECT)
        pybullet_dir = os.path.join(get_base_directory(), 'pybullet')
        franka_panda_dir = os.path.join(pybullet_dir, 'franka_panda')
        p.loadURDF(os.path.join(pybullet_dir, 'plane.urdf'), physicsClientId=self.my_id)
        p.setRealTimeSimulation(0, physicsClientId=self.my_id)
        p.setPhysicsEngineParameter(numSolverIterations=300, physicsClientId=self.my_id)
        p.setPhysicsEngineParameter(numSubSteps=10, physicsClientId=self.my_id)
        # load the robot
        self.robot = p.loadURDF(os.path.join(franka_panda_dir, 'panda_arm_hand.urdf'), self.robot_base,
                                useFixedBase=True, physicsClientId=self.my_id)
        # init all joints info
        self._number_of_all_joints = p.getNumJoints(self.robot, physicsClientId=self.my_id)
        joints_info = self._get_joints_properties()
        self._joint_names, self._joint_types, self._joints_lower_bounds, self._joints_upper_bounds = zip(*joints_info)
        # externalize only controlable joints
        enumerated_controlled_joint_info = [
            (i, joint_name, joint_type, lower_bound, upper_bound)
            for i, (joint_name, joint_type, lower_bound, upper_bound) in enumerate(joints_info)
            if joint_type != p.JOINT_FIXED
        ]
        res = list(zip(*enumerated_controlled_joint_info))
        self._external_to_internal_joint_index = res[0]
        self.joint_names = res[1]
        self.joint_types = res[2]
        self.joints_lower_bounds = res[3]
        self.joints_upper_bounds = res[4]
        self.number_of_joints = len(self._external_to_internal_joint_index)

        # objects in the world
        self.objects = set()

        # camera
        self.set_camera(3.5, 135, -20, [0., 0., 0.])

    def add_obstacles(self, obstacles_definitions_list):
        obstacles_definitions_list = [float(x) for x in obstacles_definitions_list]
        z_offset = self.robot_base[2]
        self.add_box([0.15, 0.15, z_offset - 0.15], [0.0, 0.0, z_offset / 2.])
        line_index = 0
        while line_index <= len(obstacles_definitions_list) - 6:
            sides = obstacles_definitions_list[line_index:line_index + 3]
            position = obstacles_definitions_list[line_index + 3:line_index + 6]
            self.add_box(sides, position)
            line_index += 6

    def set_camera(self, camera_distance, camera_yaw, camera_pitch, looking_at=(0., 0., 0.)):
        if self.use_ui:
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, looking_at, physicsClientId=self.my_id)

    def get_link_poses(self):
        link_poses = []
        for i in range(self._number_of_all_joints):
            state = p.getLinkState(self.robot, i, physicsClientId=self.my_id)
            position = state[4]
            orientation = state[5]
            link_poses.append((position, orientation))
        return link_poses

    def reset_robot(self):
        self.change_robot_joints([0.0 for _ in range(len(self._external_to_internal_joint_index))])

    def change_robot_joints(self, joints):
        assert len(joints) == len(self._external_to_internal_joint_index)
        for virtual_joint_index in range(len(joints)):
            joint_index = self._external_to_internal_joint_index[virtual_joint_index]
            p.resetJointState(self.robot, joint_index, targetValue=joints[virtual_joint_index], targetVelocity=0.0, physicsClientId=self.my_id)

    def get_robot_state(self):
        joint_position_velocity_pairs = [
            (t[0], t[1]) for t in p.getJointStates(self.robot, self._external_to_internal_joint_index, physicsClientId=self.my_id)
        ]
        return list(zip(*joint_position_velocity_pairs))

    def single_step_move_joint_by_position(self, joint_index, target_position, maintain_positions_other_joints=True):
        if maintain_positions_other_joints:
            target_positions = list(self.get_robot_state()[0])
            target_positions[joint_index] = target_position
            self.single_step_move_all_joints_by_position(target_positions)
        else:
            real_joint = self._external_to_internal_joint_index[joint_index]
            p.setJointMotorControl2(self.robot, real_joint, p.POSITION_CONTROL, targetPosition=target_position, force=1., physicsClientId=self.my_id)
        return self.simulation_step()

    def single_step_move_all_joints_by_position(self, target_positions):
        assert len(target_positions) == self.number_of_joints
        p.setJointMotorControlArray(
            self.robot, self._external_to_internal_joint_index, p.POSITION_CONTROL, targetPositions=target_positions, forces=[1.] * self.number_of_joints, physicsClientId=self.my_id)
        return self.simulation_step()

    def reach_joint_position(
            self, joint_index, target_position, max_steps, maintain_positions_other_joints=True, stop_on_collision=True
    ):
        trajectory = [(self.get_robot_state(), self.is_collision())]
        while len(trajectory) <= max_steps:
            last_state = trajectory[-1]
            current_joints = last_state[0][0]
            if np.abs(target_position - current_joints[joint_index]) < self.position_sensitivity:
                # close enough
                break
            if stop_on_collision and last_state[1]:
                # collision detected
                break
            trajectory.append(self.single_step_move_joint_by_position(
                joint_index, target_position, maintain_positions_other_joints))
        return trajectory

    def reach_joint_positions(
            self, target_position, max_steps, stop_on_collision=True
    ):
        trajectory = [(self.get_robot_state(), self.is_collision())]
        while len(trajectory) <= max_steps:
            last_state = trajectory[-1]
            current_joints = last_state[0][0]
            if self.is_close(current_joints, target_position):
                # close enough
                break
            if stop_on_collision and last_state[1]:
                # collision detected
                break
            trajectory.append(self.single_step_move_all_joints_by_position(target_position))
        return trajectory

    def _get_joints_properties(self):
        joints_info = [p.getJointInfo(self.robot, i, physicsClientId=self.my_id) for i in range(self._number_of_all_joints)]
        joints_info = [(t[1], t[2], t[8], t[9]) for t in joints_info]
        return joints_info

    def simulation_step(self):
        # note: if we want to activate gravity this is the place:
        # p.setGravity(0, 0, -10.)
        p.stepSimulation(physicsClientId=self.my_id)
        return self.get_robot_state(), self.is_collision()

    def is_collision(self):
        return len([
            contact_point for contact_point in p.getContactPoints(self.robot, physicsClientId=self.my_id) if contact_point[8] < 0.0
        ]) > 0

    def add_sphere(self, radius, base_position, mass=0.):
        collision_sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.my_id)
        visual_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.my_id)
        sphere = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_sphere,
                                   baseVisualShapeIndex=visual_sphere, basePosition=base_position, physicsClientId=self.my_id)
        self.objects.add(sphere)
        return sphere

    def add_box(self, sides, base_position, mass=0.):
        collision_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=sides, physicsClientId=self.my_id)
        visual_box = p.createVisualShape(p.GEOM_BOX, halfExtents=sides, physicsClientId=self.my_id)
        box = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_box,
                                   baseVisualShapeIndex=visual_box, basePosition=base_position, physicsClientId=self.my_id)
        self.objects.add(box)
        return box

    def remove_object(self, obj):
        p.removeBody(obj, physicsClientId=self.my_id)
        self.objects.remove(obj)

    def is_close(self, state1, state2=None):
        if state2 is None:
            state2 = self.get_robot_state()[0]
        return np.linalg.norm(np.array(state1) - np.array(state2)) < self.position_sensitivity

if __name__ == '__main__':
    panda_scene_manager = PandaSceneManager(use_ui=False)
    panda_scene_manager.reset_robot()
    # joint_positions, joint_velocities = panda_scene_manager.get_robot_state()
    # print(joint_positions, joint_velocities)
    # joint_positions, joint_velocities = panda_scene_manager.single_step_move_joint_by_position(
    #     0, panda_scene_manager.joints_upper_bounds[0], True)
    # print(joint_positions, joint_velocities)
    # joint_positions, joint_velocities = panda_scene_manager.single_step_move_joint_by_position(
    #     0, panda_scene_manager.joints_upper_bounds[0], False)
    # print(joint_positions, joint_velocities)
    # joint_positions, joint_velocities = panda_scene_manager.single_step_move_all_joints_by_position(
    #     panda_scene_manager.joints_upper_bounds)
    # print(joint_positions, joint_velocities)
    # panda_scene_manager.reset_robot()
    # trajectory = panda_scene_manager.reach_joint_position(0, panda_scene_manager.joints_upper_bounds[0], 100, True)
    # for t in trajectory:
    #     print(t)
    # print(len(trajectory))
    # panda_scene_manager.reset_robot()
    # trajectory = panda_scene_manager.reach_joint_position(0, panda_scene_manager.joints_upper_bounds[0], 100, False)
    # for t in trajectory:
    #     print(t)
    # print(len(trajectory))

    # panda_scene_manager.add_sphere(0.3, [0.4, 0.4, 0.4])

    scenario = 'panda_easy'
    # scenario = 'panda_hard'
    obstacles_definitions_list = []
    with open(os.path.join(get_base_directory(), 'scenario_params', scenario, 'params.pkl'), 'r') as params_file:
        obstacles_definitions_list = params_file.readlines()
    panda_scene_manager.add_obstacles(obstacles_definitions_list)

    # # easy
    # panda_scene_manager.add_box([0.5, 0.04, 0.9], [0.75, 0.0, 0.5 + z_offset])
    #
    # # hard
    # panda_scene_manager.add_box([0.5, 0.5, 0.01], [0.0, 0.8, 0.65 + z_offset])
    # panda_scene_manager.add_box([0.5, 0.5, 0.01], [0.0, -0.8, 0.65 + z_offset])
    #
    # panda_scene_manager.add_box([0.5, 0.5, 0.01], [0.8, 0.0, 0.65 + z_offset])
    # panda_scene_manager.add_box([0.5, 0.5, 0.01], [-0.8, 0.0, 0.65 + z_offset])

    def go_random_start_goal(initial_camera, steps=200, collisions_to_stop=10):
        # start somewhere
        start = np.random.uniform(panda_scene_manager.joints_lower_bounds, panda_scene_manager.joints_upper_bounds)
        # try to reach the start
        panda_scene_manager.change_robot_joints(start)
        panda_scene_manager.simulation_step()
        traj = panda_scene_manager.reach_joint_positions(start, 100)
        (start_joints, start_velocities), is_collision = traj[-1]
        if np.linalg.norm(start - start_joints) > 0.0001:
            print('failed to reach start')
            return 0
        if is_collision:
            print('started in collision')
            return 0
        collision_count = 0
        target = np.random.uniform(panda_scene_manager.joints_lower_bounds, panda_scene_manager.joints_upper_bounds)
        for i in range(steps):
            new_state, is_collision = panda_scene_manager.single_step_move_all_joints_by_position(target)
            panda_scene_manager.set_camera(3.5, i + initial_camera / 10, -20, [0., 0., 0.])
            time.sleep(0.01)
            if is_collision:
                collision_count += 1
            if collision_count == collisions_to_stop:
                print('movement in collision')
                return i
        print('random motion successful')
        return steps

    i = 0
    while(True):
        i += go_random_start_goal(i)

