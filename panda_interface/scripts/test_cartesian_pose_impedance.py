#!/usr/bin/env python

import sys
import copy
import rospy
import numpy as np

from panda_interface.arm import PandaArmInterface


if __name__ == '__main__':
    rospy.init_node('panda_client')
    config = {}
    rate = rospy.Rate(20)
    panda = PandaArmInterface(config)
    pos = panda._cartesian_pose.position
    pos[2] += 0.05
    rospy.sleep(1.)
    panda.exec_cartesian_pose_impedance_cmd(pos)

