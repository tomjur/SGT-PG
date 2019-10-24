import os

import pybullet as p
import numpy as np

from log_utils import print_and_log
from path_helper import get_base_directory


class PandaSceneManager:
    # for reference see
    # https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

    def __init__(self, use_ui=True, robot_position=(0., 0., 0.3), position_sensitivity=0.005,
                 collision_penetration=0.0001):
        self.use_ui = use_ui
        self.position_sensitivity = position_sensitivity
        self.collision_penetration = collision_penetration

        self.robot_base = robot_position
        # setup pybullet
        self.my_id = p.connect(p.GUI if use_ui else p.DIRECT)

        self.robot, self.objects = self.reset_simulation()

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

    def reset_simulation(self):
        p.resetSimulation(physicsClientId=self.my_id)
        pybullet_dir = os.path.join(get_base_directory(), 'pybullet')
        franka_panda_dir = os.path.join(pybullet_dir, 'franka_panda')
        p.loadURDF(os.path.join(pybullet_dir, 'plane.urdf'), physicsClientId=self.my_id)
        p.setRealTimeSimulation(0, physicsClientId=self.my_id)
        p.setPhysicsEngineParameter(numSolverIterations=300, physicsClientId=self.my_id)
        p.setPhysicsEngineParameter(numSubSteps=10, physicsClientId=self.my_id)
        # load the robot
        robot = p.loadURDF(os.path.join(franka_panda_dir, 'panda_arm_hand_modified_2.urdf'), self.robot_base,
                           useFixedBase=True, physicsClientId=self.my_id,
                           flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        p.setCollisionFilterPair(robot, robot, 6, 8, 0, physicsClientId=self.my_id)

        # camera
        self.set_camera(3.5, 135, -20, [0., 0., 0.])

        return robot, set()

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

    def change_robot_joints(self, joints):
        assert len(joints) == len(self._external_to_internal_joint_index)
        for virtual_joint_index in range(len(joints)):
            joint_index = self._external_to_internal_joint_index[virtual_joint_index]
            p.resetJointState(
                self.robot, joint_index, targetValue=joints[virtual_joint_index], targetVelocity=0.0,
                physicsClientId=self.my_id
            )

    def get_robot_state(self):
        joint_position_velocity_pairs = [
            (t[0], t[1])
            for t in p.getJointStates(self.robot, self._external_to_internal_joint_index, physicsClientId=self.my_id)
        ]
        return list(zip(*joint_position_velocity_pairs))

    def set_movement_target(
            self, target_joints,
            max_forces=(500., 500., 500., 500., 500., 500., 500., 10., 10.),
            position_coeff=(0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
            velocity_coeff=(1., 1., 1., 1., 1., 1., 1., 1., 1.)
    ):
        assert len(target_joints) == self.number_of_joints
        p.setJointMotorControlArray(
            self.robot, self._external_to_internal_joint_index, p.POSITION_CONTROL, targetPositions=target_joints,
            forces=max_forces, positionGains=position_coeff, velocityGains=velocity_coeff, physicsClientId=self.my_id
        )

    def _get_joints_properties(self):
        joints_info = [p.getJointInfo(self.robot, i, physicsClientId=self.my_id) for i in range(self._number_of_all_joints)]
        joints_info = [(t[1], t[2], t[8], t[9]) for t in joints_info]
        return joints_info

    def simulation_step(self):
        # note: if we want to activate gravity this is the place:
        # p.setGravity(0, 0, -10.)
        p.stepSimulation(physicsClientId=self.my_id)
        return self.get_robot_state(), self.is_collision()

    def get_collisions(self):
        return [
            contact
            for contact in p.getContactPoints(self.robot, physicsClientId=self.my_id) if contact[8] < -0.0001
        ]

    def is_collision(self):
        collisions = self.get_collisions()
        return len(collisions) > 0

    def add_sphere(self, radius, base_position, mass=0.):
        collision_sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.my_id)
        visual_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.my_id)
        sphere = p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=collision_sphere, baseVisualShapeIndex=visual_sphere,
            basePosition=base_position, physicsClientId=self.my_id
        )
        self.objects.add(sphere)
        return sphere

    def add_box(self, sides, base_position, mass=0.):
        collision_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=sides, physicsClientId=self.my_id)
        visual_box = p.createVisualShape(p.GEOM_BOX, halfExtents=sides, physicsClientId=self.my_id)
        box = p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=collision_box, baseVisualShapeIndex=visual_box,
            basePosition=base_position, physicsClientId=self.my_id
        )
        self.objects.add(box)
        return box

    def remove_object(self, obj):
        p.removeBody(obj, physicsClientId=self.my_id)
        self.objects.remove(obj)

    def is_close(self, state1, state2=None):
        if state2 is None:
            state2 = self.get_robot_state()[0]
        return np.linalg.norm(np.array(state1) - np.array(state2)) < self.position_sensitivity

    def is_moving(self):
        velocities = self.get_robot_state()[1]
        if np.linalg.norm(velocities) > 0.1:
            return True

    def walk_between_waypoints(self, start, end, max_waypoint_sensetivity_intervals=5000):
        # partition to waypoints
        waypoints = self._get_waypoints(start, end, max_waypoint_sensetivity_intervals)
        # we want to get is_goal_valid by moving in the "segment" [end_, end_]
        waypoints.append(end)

        # check each waypoint
        sum_free = 0.0
        sum_collision = 0.0
        is_start_valid, is_goal_valid = None, None

        for waypoint_index in range(len(waypoints) - 1):
            is_start_waypoint_valid, free_length, collision_length = self._walk_small_segment(
                waypoints[waypoint_index], waypoints[waypoint_index + 1]
            )
            if waypoint_index == 0:
                is_start_valid = is_start_waypoint_valid
            if waypoint_index == len(waypoints) - 2:
                is_goal_valid = is_start_waypoint_valid
            sum_free += free_length
            sum_collision += collision_length

        return is_start_valid, is_goal_valid, sum_free, sum_collision

    def _get_waypoints(self, start, end, max_waypoint_sensetivity_intervals):
        max_step = self.position_sensitivity * max_waypoint_sensetivity_intervals
        initial_distance = np.linalg.norm(end - start)
        num_steps = int(np.ceil(initial_distance / max_step))

        direction = end - start
        direction = direction / np.linalg.norm(direction)
        waypoints = [start + step*direction for step in np.linspace(0.0, initial_distance, num_steps + 1)]
        assert self.is_close(start, waypoints[0])
        assert self.is_close(end, waypoints[-1])
        return waypoints

    def _walk_small_segment(self, start, end):
        max_steps = 5000
        motionless_max_steps = max_steps/10
        segment_length = np.linalg.norm(start - end)
        if not self.is_close(start):
            self.change_robot_joints(start)
            self.simulation_step()
        is_start_valid = self.is_close(start) and not self.is_collision()
        if not is_start_valid:
            # if the segment is not free
            return is_start_valid, 0.0, segment_length
        is_free = True
        self.set_movement_target(end)
        steps = 0
        motionless_steps = 0
        while not (self.is_close(end) and not self.is_moving()):
            (current_joints, current_speed), is_collision = self.simulation_step()
            steps += 1
            if is_collision:
                is_free = False
                break
            if steps == max_steps:
                distance_to_target = np.linalg.norm(np.array(current_joints) - np.array(end))
                speed = np.linalg.norm(np.array(current_speed))
                print_and_log('segment took too long, aborting for collision. distance to target {}, speed {}'.format(
                    distance_to_target, speed
                ))
                print_and_log('start {}'.format(start.tolist()))
                print_and_log('end {}'.format(end.tolist()))
                print_and_log('current {}'.format(np.array(current_joints).tolist()))
                print_and_log('')
                is_free = False
                break
            if motionless_steps == motionless_max_steps:
                distance_to_target = np.linalg.norm(np.array(current_joints) - np.array(end))
                speed = np.linalg.norm(np.array(current_speed))
                print_and_log('segment motionless, aborting for collision. distance to target {}, speed {}'.format(
                    distance_to_target, speed
                ))
                print_and_log('start {}'.format(start.tolist()))
                print_and_log('end {}'.format(end.tolist()))
                print_and_log('current {}'.format(np.array(current_joints).tolist()))
                print_and_log('')
                is_free = False
                break
            if not self.is_moving():
                motionless_steps += 1
            else:
                motionless_steps = 0
        if is_free:
            free_length = segment_length
            collision_length = 0.0
        else:
            free_length = 0.0
            collision_length = segment_length
        return is_start_valid, free_length, collision_length
