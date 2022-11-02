#!/usr/bin/env python

import sys
import copy
import rospy

from panda_interface.arm import PandaArmInterface


if __name__ == '__main__':
    rospy.init_node('panda_client')
    config = {}
    panda = PandaArmInterface(config)
    panda.move_to_neutral()
    print(panda.tip_effort(in_base_frame=False))
