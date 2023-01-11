#!/usr/bin/env python

import sys
import copy
import rospy

from panda_interface.arm import PandaArmInterface


if __name__ == '__main__':
    rospy.init_node('panda_client')
    config = {}
    panda = PandaArmInterface(config)
    print(panda.tip_pose().position)
    # while True:
    #     print('hi')
    #     rospy.sleep(0.1)
