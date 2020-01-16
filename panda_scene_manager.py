import os
import time

import pybullet as p
import numpy as np

from path_helper import get_base_directory, get_params_from_scenario


class PandaSceneManager:
    # for reference see
    # https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

    def __init__(
            self, use_ui=True, robot_position=(0., 0., 0.3), position_sensitivity=0.005, velocity_sensitivity=0.0001,
            obstacle_definitions=None
    ):
        self.use_ui = use_ui
        self.position_sensitivity = position_sensitivity
        self.velocity_sensitivity = velocity_sensitivity

        self.obstacle_definitions = [] if obstacle_definitions is None else obstacle_definitions

        self.robot_base = robot_position
        # setup pybullet
        self.my_id = p.connect(p.GUI if use_ui else p.DIRECT)

        self.robot, self.objects = None, set()
        self.reset_simulation()

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
        self.robot = p.loadURDF(
            os.path.join(franka_panda_dir, 'panda_arm_hand_modified_2.urdf'), self.robot_base, useFixedBase=True,
            physicsClientId=self.my_id, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        )
        # p.setTimeStep(1. / 480., physicsClientId=self.my_id)
        p.setCollisionFilterPair(self.robot, self.robot, 6, 8, 0, physicsClientId=self.my_id)

        # camera
        self.set_camera(3.5, 135, -20, [0., 0., 0.])
        self._add_obstacles()

    def _add_obstacles(self):
        obstacles_definitions_list = [float(x) for x in self.obstacle_definitions]
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
        self.reset_simulation()
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
            # target_velocities=(0., 0., 0., 0., 0., 0., 0., 0., 0.),
            # max_forces=(500., 500., 500., 500., 500., 500., 500., 10., 10.),
            # position_coeff=(0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
            # velocity_coeff=(1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000.)
    ):
        assert len(target_joints) == self.number_of_joints
        # assert len(target_joints) == len(target_velocities) == self.number_of_joints
        # assert len(max_forces) == len(position_coeff) == len(velocity_coeff) == self.number_of_joints
        p.setJointMotorControlArray(
            self.robot, self._external_to_internal_joint_index, p.POSITION_CONTROL,
            targetPositions=target_joints,
            # targetVelocities=target_velocities,
            # forces=max_forces,
            # positionGains=position_coeff,
            # velocityGains=velocity_coeff,
            physicsClientId=self.my_id
        )

    def _get_joints_properties(self):
        joints_info = [
            p.getJointInfo(self.robot, i, physicsClientId=self.my_id) for i in range(self._number_of_all_joints)
        ]
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

    def is_close(self, target_joints, source_joints=None):
        distance = self.get_distance(target_joints, source_joints)
        return distance < self.position_sensitivity

    def get_distance(self, target_joints, source_joints=None):
        assert len(target_joints) == self.number_of_joints
        if source_joints is None:
            source_joints = self.get_robot_state()[0]
        assert len(source_joints) == self.number_of_joints
        return np.linalg.norm(np.array(source_joints) - np.array(target_joints))

    def is_moving(self):
        current_speed = self.get_current_speed()
        return current_speed > self.velocity_sensitivity

    def get_current_speed(self):
        velocities = self.get_robot_state()[1]
        return np.linalg.norm(velocities)

    def smooth_walk(self, goal, max_target_distance=None, sensitivity=None):
        goal_ = np.array(goal)
        assert sensitivity is None or sensitivity > 0.
        if sensitivity is None:
            sensitivity = 10. * self.position_sensitivity
        assert max_target_distance is None or max_target_distance > 0.
        assert max_target_distance > sensitivity

        original_start = np.array(self.get_robot_state()[0])

        success = self._execute_smooth_walk(goal_, max_target_distance, sensitivity)

        if success:
            goal_start_direction = goal_ - original_start
            original_distance = np.linalg.norm(goal_start_direction)
            success_sum = original_distance
            collision_sum = 0.
        else:
            current_joints = np.array(self.get_robot_state()[0])
            success_sum = np.linalg.norm(current_joints - original_start)
            collision_sum = np.linalg.norm(current_joints - goal_)
        return success_sum, collision_sum

    def _execute_smooth_walk(self, goal, max_target_distance, sensitivity):
        steps_counter = 0
        current_joints = np.array(self.get_robot_state()[0])
        while not (self.is_close(goal) and not self.is_moving()):
            direction = goal - current_joints
            distance = np.linalg.norm(direction)
            direction = direction / distance
            if max_target_distance is not None and distance > 2 * max_target_distance:
                move_target = current_joints + max_target_distance * direction
            elif distance >= 2 * sensitivity:
                move_target = (current_joints + goal) * 0.5
            elif distance >= sensitivity:
                move_target = current_joints + sensitivity * direction
            else:
                move_target = goal
            self.set_movement_target(move_target)
            (current_joints, _), is_collision = self.simulation_step()
            current_joints = np.array(current_joints)
            if is_collision:
                return False
            if steps_counter == 5000:
                return False
            steps_counter += 1
        return True

    def walk_between_waypoints(
            self, start, end, max_waypoint_sensetivity_intervals=20,
            teleport_between_waypoints=True, time_between_frames=0.0
    ):
        assert len(start) == len(end) == self.number_of_joints
        # partition to waypoints
        waypoints = self._get_waypoints(np.array(start), np.array(end), max_waypoint_sensetivity_intervals)
        # we want to get is_goal_valid by moving in the "segment" [end_, end_]
        waypoints.append(end)

        # check each waypoint
        sum_free = 0.0
        sum_collision = 0.0
        is_start_valid, is_goal_valid = None, None

        visited_by_waypoint = []

        for waypoint_index in range(len(waypoints) - 1):
            start_joints = waypoints[waypoint_index] if teleport_between_waypoints else None
            is_start_waypoint_valid, free_length, collision_length, visited_states = self.walk_small_segment(
                waypoints[waypoint_index+1], time_between_frames=time_between_frames,
                required_start_joints=start_joints, max_steps=500
            )
            # is_start_waypoint_valid, free_length, collision_length, _ = self.bounded_jerk_motion_model_predictive_control(
            #     waypoints[waypoint_index+1], required_start_joints=start_joints, time_between_frames=time_between_frames
            # )
            if waypoint_index == 0:
                is_start_valid = is_start_waypoint_valid
            if waypoint_index == len(waypoints) - 2:
                is_goal_valid = is_start_waypoint_valid
            sum_free += free_length
            sum_collision += collision_length

            visited_by_waypoint.append(visited_states)

        return is_start_valid, is_goal_valid, sum_free, sum_collision, visited_by_waypoint

    # def bounded_jerk_motion_model_predictive_control(
    #         self, target_joints,
    #         maximal_jerk=(100., 100., 100., 100., 100., 100., 100., 100., 100.),
    #         max_forces=(500., 500., 500., 500., 500., 500., 500., 10., 10.),
    #         kd_position_coeff=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    #         kd_velocity_coeff=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    #         time_between_frames=0., required_start_joints=None
    # ):
    #     assert len(target_joints) == len(maximal_jerk) == self.number_of_joints
    #
    #     if required_start_joints is not None:
    #         if not self.is_close(required_start_joints):
    #             self.change_robot_joints(required_start_joints)
    #             self.simulation_step()
    #
    #     # get the current robot state
    #     starting_joints, starting_velocities = self.get_robot_state()
    #     segment_length = self.get_distance(target_joints, starting_joints)
    #
    #     # get the simulation time step
    #     engine_parameters = p.getPhysicsEngineParameters(physicsClientId=self.my_id)
    #     time_step = engine_parameters['fixedTimeStep']
    #     # check if initial state is collision
    #     if self.is_collision():
    #         return False, 0.0, segment_length, [(starting_joints, starting_velocities)]
    #
    #     # these will hold the budget of steps the simulation is allowed to execute
    #     max_steps = None
    #     motionless_max_steps = None
    #
    #     motionless_steps = 0
    #     is_free = True
    #     visited_states = []
    #     while not (self.is_close(target_joints) and not self.is_moving()):
    #         visited_states.append(self.get_robot_state())
    #         joints, velocities = visited_states[-1]
    #         # the required time is the time maximal required time by any joint
    #         time_required = np.max([
    #             PandaSceneManager._find_minimal_time(
    #                 joints[i], target_joints[i], velocities[i], time_step, maximal_jerk[i])
    #             for i in range(self.number_of_joints)
    #         ])
    #         # compute the allowed budget of steps
    #         if max_steps is None:
    #             max_steps = int(time_required / time_step + 1) * 2
    #             motionless_max_steps = int(max_steps / 10.) + 1
    #         # the target of the next simulation step
    #         targets = [
    #             PandaSceneManager._compute_target_position_velocity(
    #                 joints[i], target_joints[i], velocities[i], time_step, time_required)
    #             for i in range(self.number_of_joints)
    #         ]
    #         next_joints, next_velocities = zip(*targets)
    #         self.set_movement_target(next_joints, next_joints, max_forces, kd_position_coeff, kd_velocity_coeff)
    #         # execute and observe the result
    #         (current_joints, current_speed), is_collision = self.simulation_step()
    #         if is_collision:
    #             is_free = False
    #             break
    #         if (len(visited_states) + 1) == max_steps or motionless_steps == motionless_max_steps:
    #             if (len(visited_states) + 1) == max_steps:
    #                 print_and_log('segment too long, aborting for collision. distance to target {}, speed {}'.format(
    #                     self.get_distance(target_joints, current_joints), self.get_current_speed()
    #                 ))
    #             else:
    #                 print_and_log('segment motionless, aborting for collision. distance to target {}, speed {}'.format(
    #                     self.get_distance(target_joints, current_joints), self.get_current_speed()
    #                 ))
    #             print_and_log('start configuration {}, starting speeds {}'.format(starting_joints, starting_velocities))
    #             print_and_log('end {}'.format(target_joints.tolist()))
    #             print_and_log('current configuration {}, current speeds {}'.format(current_joints, current_speed))
    #             print_and_log('')
    #             is_free = False
    #             break
    #         if not self.is_moving():
    #             motionless_steps += 1
    #         else:
    #             motionless_steps = 0
    #         if time_between_frames > 0.:
    #             time.sleep(time_between_frames)
    #     if is_free:
    #         free_length = segment_length
    #         collision_length = 0.0
    #     else:
    #         free_length = 0.0
    #         collision_length = segment_length
    #     return True, free_length, collision_length, visited_states
    #
    # @staticmethod
    # def _compute_target_position_velocity(current_joint, target_joint, current_velocity, time_step, end_time):
    #     a0 = current_joint
    #     a1 = current_velocity
    #     a2 = PandaSceneManager._compute_a2(current_joint, target_joint, current_velocity, end_time)
    #     a3 = PandaSceneManager._compute_a3(current_joint, target_joint, current_velocity, end_time)
    #     position = a3 * time_step ** 3 + a2 * time_step ** 2 + a1 * time_step + a0
    #     velocity = 3 * a3 * time_step ** 2 + 2 * a2 * time_step + a1
    #     return position, velocity
    #
    # @staticmethod
    # def _find_minimal_time(current_joint, target_joint, current_velocity, pybullet_step_time, constraint):
    #     end_time = pybullet_step_time
    #     while np.abs(6 * PandaSceneManager._compute_a3(
    #             current_joint, target_joint, current_velocity, end_time)) > constraint:
    #         end_time *= 2
    #     return end_time
    #
    # @staticmethod
    # def _compute_a2(current_joint, target_joint, current_velocity, end_time):
    #     return (3. * target_joint - 3. * current_joint - 2. * current_velocity * end_time) / (end_time ** 2)
    #
    # @staticmethod
    # def _compute_a3(current_joint, target_joint, current_velocity, end_time):
    #     return (current_velocity * end_time + 2. * current_joint - 2. * target_joint) / (end_time ** 3)

    def _get_waypoints(self, start, end, max_waypoint_sensetivity_intervals):
        max_step = self.position_sensitivity * max_waypoint_sensetivity_intervals
        direction = end - start
        initial_distance = np.linalg.norm(direction)
        if initial_distance < 0.0001:
            return [start, end]
        num_steps = int(np.ceil(initial_distance / max_step))
        direction = direction / initial_distance
        waypoints = [start + step*direction for step in np.linspace(0.0, initial_distance, num_steps + 1)]
        assert self.is_close(start, waypoints[0])
        assert self.is_close(end, waypoints[-1])
        return waypoints

    def walk_small_segment(
            self, target_joints, time_between_frames=0., required_start_joints=None, max_steps=5000,
            # max_forces=(500., 500., 500., 500., 500., 500., 500., 10., 10.),
            # position_coeff=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            # velocity_coeff=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    ):

        motionless_max_steps = max_steps/10

        assert len(target_joints) == self.number_of_joints

        if required_start_joints is not None:
            if not self.is_close(required_start_joints):
                self.change_robot_joints(required_start_joints)
                self.simulation_step()

        # get the current robot state
        starting_joints, starting_velocities = self.get_robot_state()
        segment_length = self.get_distance(target_joints, starting_joints)

        # check if initial state is collision
        if self.is_collision():
            return False, 0.0, segment_length, [(starting_joints, starting_velocities)]

        # these will hold the budget of steps the simulation is allowed to execute
        is_free = True
        visited_states = []
        motionless_steps = 0
        self.set_movement_target(
            target_joints)
            # target_joints, max_forces=max_forces, position_coeff=position_coeff, velocity_coeff=velocity_coeff)
        while not (self.is_close(target_joints) and not self.is_moving()):
            visited_states.append(self.get_robot_state())
            # execute and observe the result
            (current_joints, current_speed), is_collision = self.simulation_step()
            if is_collision:
                is_free = False
                break
            if (len(visited_states) + 1) == max_steps or motionless_steps == motionless_max_steps:
                # if (len(visited_states) + 1) == max_steps:
                #     print_and_log('segment too long, aborting for collision. distance to target {}, speed {}'.format(
                #         self.get_distance(target_joints, current_joints), self.get_current_speed()
                #     ))
                # else:
                #     print_and_log('segment motionless, aborting for collision. distance to target {}, speed {}'.format(
                #         self.get_distance(target_joints, current_joints), self.get_current_speed()
                #     ))
                # print_and_log('start configuration {}, starting speeds {}'.format(starting_joints, starting_velocities))
                # print_and_log('end {}'.format(target_joints))
                # print_and_log('current configuration {}, current speeds {}'.format(current_joints, current_speed))
                # print_and_log('')
                is_free = False
                break
            if not self.is_moving():
                motionless_steps += 1
            else:
                motionless_steps = 0
            if time_between_frames > 0.:
                time.sleep(time_between_frames)
        if is_free:
            free_length = segment_length
            collision_length = 0.0
        else:
            free_length = 0.0
            collision_length = segment_length
        return True, free_length, collision_length, visited_states

    @staticmethod
    def get_scene_manager(scenario, use_ui=False):
        obstacles_definitions_list = PandaSceneManager._get_obstacle_definitions(scenario)
        panda_scene_manager = PandaSceneManager(use_ui=use_ui, obstacle_definitions=obstacles_definitions_list)
        return panda_scene_manager

    @staticmethod
    def _get_obstacle_definitions(scenario):
        params_file = get_params_from_scenario(scenario)
        if 'no_obs' in params_file:
            obstacles_definitions_list = []
        else:
            with open(params_file, 'r') as f:
                obstacles_definitions_list = f.readlines()
        return obstacles_definitions_list
