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
    for _ in range(20):
        panda.exec_cartesian_velocity_cmd(np.array([0, 0, 0.1]))
        rate.sleep()
