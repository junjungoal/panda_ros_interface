#!/usr/bin/env python

import sys
import copy
import rospy
import actionlib
from threading import Lock, Event
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import *
from std_msgs.msg import Float32MultiArray
import time

import franka_gripper
import franka_gripper.msg
from franka_msgs.msg import FrankaState, ErrorRecoveryGoal, ErrorRecoveryAction, ErrorRecoveryActionGoal
from franka_msgs.srv import SetFullCollisionBehavior, SetCartesianImpedance, SetJointImpedance
from rospy_message_converter import message_converter

from controller_manager_msgs.srv import ListControllers
from controller_manager_msgs.srv import SwitchController

import moveit_commander
from moveit_commander.conversions import pose_to_list
import moveit_msgs.msg
from moveit_msgs.msg import Constraints, OrientationConstraint

import numpy as np
from math import pi
from panda_interface.utils.general_utils import AttrDict, ParamDict
from panda_interface.utils.transform_utils import mat2quat, quat2mat, quat_multiply
from panda_interface.utils.ros_utils import create_pose_msg
from panda_interface.move_group_interface import PandaMoveGroupInterface
from panda_interface.collision_behaviour_interface import CollisionBehaviourInterface
from panda_interface.gripper import GripperInterface

class PandaArmInterface(object):
    def __init__(self, config=AttrDict()):
        self._hp = self._default_hparams().overwrite(config)

        self._joint_names = rospy.get_param("/franka_control/joint_names")
        self._joint_limits = self._get_joint_limits()
        # self._neutral_pose_joints = self._get_neutral_pose()
        self._neutral_pose_joints = dict(zip(self._joint_names, self._get_neutral_pose()))

        self._movegroup_interface = PandaMoveGroupInterface()
        self._collision_behaviour_interface = CollisionBehaviourInterface()
        self._collision_behaviour_interface.set_ft_contact_collision_behaviour()
        self._gripper = GripperInterface()

        rospy.wait_for_service('/controller_manager/switch_controller')
        self.switcher_srv = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

        self.franka_state = None
        self.error_recovery_client = actionlib.SimpleActionClient('/franka_control/error_recovery',
                                                                  ErrorRecoveryAction)

        # Waits until the action server has started up and started
        # listening for goals.
        print ("Waiting for error recovery server in franka_control")
        self.error_recovery_client.wait_for_server()
        print ("Found error recovery server")
        self.recover_from_errors()

        rospy.loginfo("Waiting for Franka States")
        self.state_subscriber = rospy.Subscriber('/franka_state_controller/franka_states',
                                                 FrankaState,
                                                 self._update_states)
        rospy.loginfo("Connected to Franka States")

        self.joint_subscriber = rospy.Subscriber('/franka_state_controller/joint_states',
                                                 JointState,
                                                 self._update_joints)

        self._cartesian_pose = None
        self._cartesian_effort = None
        self._cartesian_velocity = None
        self._timestamp_secs = None
        self._joint_angles = AttrDict()
        self._joint_velocity = AttrDict()
        self._joint_effort = AttrDict()

        if self.is_in_controller_list('cartesian_velocity_controller'):
            self.switch_controllers('cartesian_velocity_controller')
            rospy.loginfo("Waiting for current target velocity")
            self.cmd_vel_pub = rospy.Publisher("/franka_control/target_velocity", Twist, queue_size=1)
            rospy.loginfo("Connected to current trarget velocity")

        if self.is_in_controller_list('cartesian_pose_impedance_controller'):
            self.switch_controllers('cartesian_pose_impedance_controller')
            rospy.loginfo("Waiting for cartesian pose impedance publisher")
            self.cmd_pose_impedance_pub = rospy.Publisher("/cartesian_impedance_controller/desired_pose", PoseStamped, queue_size=10)
            rospy.sleep(1.)
            rospy.loginfo("Connected to cartesian pose impedance publisher")
        self.switch_controllers('position_joint_trajectory_controller')



    def _get_neutral_pose(self):
        joint_states = []
        for name in self._joint_names:
            joint_states.append(rospy.get_param("/franka_control/neutral_pose/{}".format(name)))
        return np.array(joint_states)

    def _get_joint_limits(self):
        """
        Get joint limits as defined in ROS parameter server
        (/franka_control/joint_config/joint_velocity_limit)
        :return: Joint limits for each joints
        :rtype: franka_core_msgs.msg.JointLimits
        """
        keys = ['position_upper', 'position_lower', 'velocity', 'accel', 'effort']
        lims = AttrDict()
        for key in keys:
            lims[key] = []
        vel_lim = rospy.get_param("/franka_control/joint_config/joint_velocity_limit")
        pos_min_lim = rospy.get_param("/franka_control/joint_config/joint_position_limit/lower")
        pos_max_lim = rospy.get_param("/franka_control/joint_config/joint_position_limit/upper")
        eff_lim = rospy.get_param("/franka_control/joint_config/joint_effort_limit")
        acc_lim = rospy.get_param("/franka_control/joint_config/joint_acceleration_limit")

        for i in range(len(self._joint_names)):
            lims.position_upper.append(pos_max_lim[self._joint_names[i]])
            lims.position_lower.append(pos_min_lim[self._joint_names[i]])
            lims.velocity.append(vel_lim[self._joint_names[i]])
            lims.accel.append(acc_lim[self._joint_names[i]])
            lims.effort.append(eff_lim[self._joint_names[i]])

        return lims


    def _default_hparams(self):
        default_dict = ParamDict({
            'group_name': 'panda_arm',
            'hand_group_name': 'hand',
            'load_gripper': True,
            'ee_safety_zone': [[0.30, 0.76], [-0.25, 0.25], [0.01, 0.4]],
        })
        return default_dict

    def ee_inside_safety_zone(self, xyz):
        return xyz[0] >= self._hp.ee_safety_zone[0][0] and xyz[0] <= self._hp.ee_safety_zone[0][1] and \
            xyz[1] >= self._hp.ee_safety_zone[1][0] and xyz[1] <= self._hp.ee_safety_zone[1][1] and \
            xyz[2] >= self._hp.ee_safety_zone[2][0] and xyz[2] <= self._hp.ee_safety_zone[2][1]

    def _update_states(self, msg):
        cart_pose_mat = np.asarray(msg.O_T_EE).reshape(4, 4, order='F')

        self._cartesian_pose = AttrDict(
            position=cart_pose_mat[:3, 3],
            orientation=mat2quat(cart_pose_mat),
            ori_mat=cart_pose_mat[:3, :3],
        )


        self._cartesian_velocity = AttrDict(
            linear=np.array(msg.O_dP_EE[:3]),
            angular=np.array([msg.O_dP_EE[3:]])
        )

        self._cartesian_effort = AttrDict(
            force=msg.O_F_ext_hat_K[:3],
            torque=msg.O_F_ext_hat_K[3:]
        )

        self._stiffness_frame_effort = AttrDict(
            force=msg.K_F_ext_hat_K[:3],
            torque=msg.K_F_ext_hat_K[3:]
        )

        self._cartesian_contact = msg.cartesian_contact
        self._cartesian_collision = msg.cartesian_collision

        self._joint_contact = msg.joint_contact
        self._joint_collision = msg.joint_collision

        # self._joint_inertia = np.array(msg.mass_matrix).reshape(7, 7, order='F')

        self.q_d = msg.q_d
        self.dq_d = msg.dq_d

        self._F_T_NE = msg.F_T_NE # should be constant normally
        # self._NE_T_EE = msg.NE_T_EE
        self._F_T_EE = msg.F_T_EE

        # self._gravity = np.array(msg.gravity)
        # self._coriolis = np.array(msg.coriolis)
        self._errors = message_converter.convert_ros_message_to_dictionary(
            msg.current_errors
        )
        self.franka_state = msg

    def _update_joints(self, msg):
        for i, name in enumerate(self._joint_names):
            if name in self._joint_names:
                self._joint_angles[name] = msg.position[i]
                self._joint_velocity[name] = msg.velocity[i]
                self._joint_effort[name] = msg.effort[i]


    def joint_angle(self, joint_name):
        return self._joint_angles[joint_name]

    def joint_angles(self):
        return copy.deepcopy(self._joint_angles)

    def joint_ordered_angles(self):
        return np.array([self._joint_angles[name] for name in self._joint_names])

    def joint_velocity(self, joint_name):
        return self._joint_velocity[joint_name]

    def joint_velocities(self):
        return copy.deepcopy(self._joint_velocity)

    def joint_effort(self, joint_name):
        return self._joint_effort[joint_name]

    def joint_efforts(self):
        return copy.deepcopy(self._joint_effort)

    def tip_pose(self):
        return copy.deepcopy(self._cartesian_pose)

    def tip_velocity(self):
        return copy.deepcopy(self._cartesian_velocity)

    def tip_effort(self, in_base_frame=True):
        return copy.deepcopy(self._cartesian_effort) if in_base_frame else copy.deepcopy(self._stiffness_frame_effort)

    def set_joint_position_speed(self, speed=0.3):
        """
        Set ratio of max joint speed to use during joint position
        moves (only for move_to_joint_positions).
        Set the proportion of maximum controllable velocity to use
        during joint position control execution. The default ratio
        is `0.3`, and can be set anywhere from [0.0-1.0] (clipped).
        Once set, a speed ratio will persist until a new execution
        speed is set.
        :type speed: float
        :param speed: ratio of maximum joint speed for execution
                      default= 0.3; range= [0.0-1.0]
        """
        if speed > 0.3:
            rospy.logwarn("{}: Setting speed above 0.3 could be risky!! Be extremely careful.".format(
                self.__class__.__name__))
        if self._movegroup_interface:
            self._movegroup_interface.set_velocity_scale(speed * 2)
        self._speed_ratio = speed

    def set_gripper_speed(self, speed):
        if self._gripper:
            self._gripper.set_velocity(speed)

    def exec_gripper_cmd(self, pos, force=None):
        """
        Move gripper joints to the desired width (space between finger joints), while applying
        the specified force (optional)
        :param pos: desired width [m]
        :param force: desired force to be applied on object [N]
        :type pos: float
        :type force: float
        :return: True if command was successful, False otherwise.
        :rtype: bool
        """
        if self._gripper is None:
            return

        width = min(self._gripper.MAX_WIDTH, max(self._gripper.MIN_WIDTH, pos))

        if force:
            holding_force = min(
                max(self._gripper.MIN_FORCE, force), self._gripper.MAX_FORCE)

            return self._gripper.grasp(width=width, force=holding_force)

        else:
            return self._gripper.move_joints(width)

    def exec_cartesian_pose_impedance_cmd(self, pos, ori=None):
        running = self.controller_is_running('cartesian_pose_impedance_controller')
        if not running:
            print ("Switching to position control")
            resp = self.switch_controllers('cartesian_pose_impedance_controller')

        pose = PoseStamped()
        pose.header.stamp = rospy.Time(0)
        if ori is None:
            ori = self._cartesian_pose['orientation']

        pose.pose.orientation.x = ori[0]
        pose.pose.orientation.y = ori[1]
        pose.pose.orientation.z = ori[2]
        pose.pose.orientation.w = ori[3]

        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]

        current_pose = self.tip_pose()
        if np.abs(current_pose.position[0] - pos[0]) > 0.1 or np.abs(current_pose.position[1]-pos[1]) > 0.1 or np.abs(current_pose.position[2] - pos[2]) > 0.1:
            raise ValueError('Invalid position value')


        self.cmd_pose_impedance_pub.publish(pose)

    def exec_cartesian_velocity_cmd(self, twist):
        if not isinstance(twist, Twist):
            new_twist = Twist()
            new_twist.linear.x = twist[0]
            new_twist.linear.y = twist[1]
            new_twist.linear.z = twist[2]
            twist = new_twist

        running = self.controller_is_running('cartesian_velocity_controller')
        if not running:
            print ("Switching to position control")
            resp = self.switch_controllers('cartesian_velocity_controller')
        #
        # Clip x-velocity if outsize the safety zone and request is to further violate it
        #
        if self._cartesian_pose.position[0] < self._hp.ee_safety_zone[0][0] and twist.linear.x < 0:
            twist.linear.x = 0.0

        if self._cartesian_pose.position[0] > self._hp.ee_safety_zone[0][1] and twist.linear.x > 0:
            twist.linear.x = 0.0

        #
        # Clip y-velocity if outsize the safety zone and request is to further violate it
        #
        if self._cartesian_pose.position[1] < self._hp.ee_safety_zone[1][0] and twist.linear.y < 0:
            twist.linear.y = 0.0

        if self._cartesian_pose.position[1] > self._hp.ee_safety_zone[1][1] and twist.linear.y > 0:
            twist.linear.y = 0.0

        #
        # Clip z-velocity if outsize the safety zone and request is to further violate it
        #
        if self._cartesian_pose.position[2] < self._hp.ee_safety_zone[2][0] and twist.linear.z < 0:
            twist.linear.z = 0.0

        if self._cartesian_pose.position[2] > self._hp.ee_safety_zone[2][1] and twist.linear.z > 0:
            twist.linear.z = 0.0

        print(twist)
        self.cmd_vel_pub.publish(twist)

    def recover_from_errors(self):
        goal = ErrorRecoveryActionGoal()
        # goal = ErrorRecoveryGoal()
        self.error_recovery_client.send_goal(goal)
        print ("Waiting for recovery goal")
        self.error_recovery_client.wait_for_result()
        print ("Done")
        return self.error_recovery_client.get_result()

    def move_to_neutral(self, timeout=15, speed=0.15):
        self.set_joint_position_speed(speed)
        self.move_to_joint_positions(self._neutral_pose_joints, timeout)

    def move_to_joint_positions(self, positions,
                                timeout=10.0,
                                threshold=0.00085,
                                test=None, use_moveit=True):

        running = self.controller_is_running('position_joint_trajectory_controller')
        if not running:
            print ("Switching to position control")
            resp = self.switch_controllers('position_joint_trajectory_controller')

        if isinstance(positions, list) or isinstance(positions, np.ndarray):
            positions = dict(zip(self._joint_names, positions))

        if use_moveit and self._movegroup_interface:
            self._movegroup_interface.go_to_joint_positions(
                [positions[n] for n in self._joint_names], tolerance=threshold
            )
        else:
            if use_moveit:
                rospy.logwarn("{}: MoveGroupInterface was not found! Using JointTrajectoryActionClient instead.".format(
                    self.__class__.__name__))
            raise NotImplementedError

        rospy.loginfo("{}: Trajectory controlling complete".format(
            self.__class__.__name__))

    def get_flange_pose(self, pos=None, ori=None):
        """
        Get the pose of flange (panda_link8) given the pose of the end-effector frame.
        .. note:: In sim, this method does nothing.
        :param pos: position of the end-effector frame in the robot's base frame, defaults to current end-effector position
        :type pos: np.ndarray, optional
        :param ori: orientation of the end-effector frame, defaults to current end-effector orientation
        :type ori: quaternion.quaternion, optional
        :return: corresponding flange frame pose in the robot's base frame
        :rtype: np.ndarray, quaternion.quaternion
        """
        if pos is None:
            pos = self._cartesian_pose['position']

        if ori is None:
            ori = self._cartesian_pose['orientation']

        # get corresponding flange frame pose using transformation matrix
        F_T_EE = np.asarray(self._F_T_EE).reshape(4, 4, order="F")
        mat = quat2mat(ori)

        new_ori = mat.dot(F_T_EE[:3,:3].T)
        new_pos = pos - new_ori.dot(F_T_EE[:3, 3])

        return new_pos, mat2quat(new_ori).astype(np.float64)

    def move_to_cartesian_delta(self, pos, ori=None, use_moveit=True):
        current_pose = self._cartesian_pose
        desired_pos = current_pose['position'] + pos
        if ori is None:
            ori = np.array([0, 0, 0, 1])
        desired_ori = quat_multiply(current_pose['orientation'], ori)
        return self.move_to_cartesian_pose(desired_pos, desired_ori, use_moveit=use_moveit)

    def move_to_cartesian_pose(self, pos, ori=None, use_moveit=True):
        if not use_moveit or self._movegroup_interface is None:
            rospy.logerr("{}: MoveGroupInterface was not found! Aborting cartesian planning.".format(
                self.__class__.__name__))
            return

        running = self.controller_is_running('position_joint_trajectory_controller')
        if not running:
            print ("Switching to position control")
            resp = self.switch_controllers('position_joint_trajectory_controller')

        if ori is None:
            ori = self._cartesian_pose['orientation']

        self._movegroup_interface.go_to_cartesian_pose(
            create_pose_msg(*self.get_flange_pose(pos, ori))
        )
        rospy.loginfo("{}: Trajectory controlling complete".format(
            self.__class__.__name__))

    def switch_controllers(self, start_controller):
        stop_controller = self.get_current_controller()
        try:
            req = {'start_controllers': [start_controller] if start_controller is not None else [],
                   'stop_controllers': [stop_controller] if stop_controller is not None else [],
                   'strictness': 2,  # 2=strict, 1=best effort
                    #'start_asap': True, # for some reason this was excluded from ROS melodic
                    #'timeout': 3,  # in seconds
            }
            resp = self.switcher_srv(**req)
            rospy.sleep(0.2)
            return resp
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def get_current_controller(self):
        controller_status = self.query_controller_status()
        for cs in controller_status.controller:
            if cs.state == 'running' and cs.name != "franka_state_controller":
                return cs.name
        # print("No controller found")

    def controller_is_running(self, name):
        controller_status = self.query_controller_status()
        for cs in controller_status.controller:
            if cs.name == name:
                return cs.state == 'running'

        return False

    def is_in_controller_list(self, name):
        controller_status = self.query_controller_status()
        for cs in controller_status.controller:
            if cs.name == name:
                return True
        return False

    def query_controller_status(self):
        rospy.wait_for_service('/controller_manager/list_controllers')
        try:
            srv = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
            resp = srv()
            return resp
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def set_collision_threshold(self, cartesian_forces=None, joint_torques=None):
        """
        Set Force Torque thresholds for deciding robot has collided.
        :return: True if service call successful, False otherwise
        :rtype: bool
        :param cartesian_forces: Cartesian force threshold for collision detection [x,y,z,R,P,Y] (robot motion stops if violated)
        :type cartesian_forces: [float] size 6
        :param joint_torques: Joint torque threshold for collision (robot motion stops if violated)
        :type joint_torques: [float] size 7
        """
        if self._collision_behaviour_interface:
            return self._collision_behaviour_interface.set_collision_threshold(joint_torques=joint_torques, cartesian_forces=cartesian_forces)
        else:
            rospy.logwarn("No CollisionBehaviourInterface object found!")


    def set_cartesian_impedance(self, cb=None):
        current = self.get_current_controller()
        self.switch_controllers(None)
        rospy.wait_for_service("/franka_control/set_cartesian_impedance")
        if cb is None:
            cb = [2000, 2000, 2000, 300, 300, 300]

        try:
            cartesian_impedance_srv = rospy.ServiceProxy("/franka_control/set_cartesian_impedance", SetCartesianImpedance)
            resp = cartesian_impedance_srv(cb)
            rospy.sleep(0.2)
            self.switch_controllers(current)
            return resp.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def set_joint_impedance(self, cb=None):
        current = self.get_current_controller()
        self.switch_controllers(None)
        rospy.wait_for_service("/franka_control/set_joint_impedance")
        if cb is None:
            cb = [3000, 3000, 3000, 2500, 2500, 2000, 2000]

        try:
            joint_impedance_srv = rospy.ServiceProxy("/franka_control/set_joint_impedance", SetJointImpedance)
            # resp = joint_impedance_srv(**cb)
            resp = joint_impedance_srv(cb)
            rospy.sleep(0.2)
            self.switch_controllers(current)
            return resp.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def is_in_safe_state(self):
        return (not self.is_in_collision_mode() and \
                not self.is_in_user_stop_mode() and \
                not self.error_in_current_state())

    def is_in_contact_mode(self):
        if self.franka_state is None:
            return False

        return any(self.franka_state.joint_contact)

    def is_in_collision_mode(self):
        if self.franka_state is None:
            return False

        return self.franka_state.robot_mode == FrankaState.ROBOT_MODE_REFLEX

    def is_in_move_mode(self):
        if self.franka_state is None:
            return False

        return self.franka_state.robot_mode == FrankaState.ROBOT_MODE_MOVE

    def is_in_idle_mode(self):
        if self.franka_state is None:
            return False

        return self.franka_state.robot_mode == FrankaState.ROBOT_MODE_IDLE

    def is_in_user_stop_mode(self):
        if self.franka_state is None:
            return False
        return self.franka_state.robot_mode == FrankaState.ROBOT_MODE_USER_STOPPED

    def error_in_current_status(self):
        return not all([e == False for e in list(self._errors.values())])

    def get_robot_status(self):
        return AttrDict(
            robot_mode=self.franka_state.robot_mode,
            robot_status=(not self.is_in_collision_mode() and not self.is_in_user_stop_mode()),
            errors=self._errors,
            error_in_current_status=self.error_in_current_status()
        )

    @property
    def gripper(self):
        return self._gripper

    @property
    def coriolis_comp(self):
        return self._coriolis

    @property
    def gravity_comp(self):
        return self._gravity

    @property
    def movegroup_interface(self):
        return self._movegroup_interface

    @property
    def joint_limits(self):
        return self._joint_limits

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def errors(self):
        return self._errors

