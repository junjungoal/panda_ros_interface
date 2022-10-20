cmake_minimum_required(VERSION 3.0.2)
project(panda_ros_controllers)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED
  COMPONENTS
  controller_interface
  dynamic_reconfigure
  franka_hw
  geometry_msgs
  hardware_interface
  message_generation
  realtime_tools
  roscpp
  pluginlib
  geometric_shapes
  pcl_ros
  pcl_conversions
  rosbag
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  rospy
)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED system filesystem date_time thread)

catkin_python_setup()

generate_messages()

generate_dynamic_reconfigure_options(
    cfg/minimal_compliance_param.cfg
)

#catkin_package(
#  LIBRARIES
#  INCLUDE_DIRS
#  CATKIN_DEPENDS
#    tf2_geometry_msgs
#  DEPENDS
#    EIGEN3
#)



catkin_package(
  LIBRARIES panda_ros_controllers
  CATKIN_DEPENDS
    controller_interface
    dynamic_reconfigure
    franka_hw
    geometry_msgs
    hardware_interface
    message_runtime
    pluginlib
    realtime_tools
    roscpp
  DEPENDS Franka
)

add_library(panda_ros_controllers
  src/cartesian_velocity_controller.cpp
  src/cartesian_pose_impedance_controller.cpp
)

add_dependencies(panda_ros_controllers
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  ${PROJECT_NAME}_generate_messages_cpp
  ${PROJECT_NAME}_gencpp
  ${PROJECT_NAME}_gencfg
)

include_directories(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)


target_link_libraries(panda_ros_controllers PUBLIC
  ${Franka_LIBRARIES}
  ${catkin_LIBRARIES}
)

target_include_directories(panda_ros_controllers SYSTEM PUBLIC
  ${Franka_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
target_include_directories(panda_ros_controllers PUBLIC
  include
)


## Installation
install(TARGETS panda_ros_controllers
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

###########
## Build ##
###########

include_directories(${THIS_PACKAGE_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})

install(PROGRAMS
  src/remap_joy_to_twist.py
  src/panda_env_test.py
  src/panda_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


## Tools
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ClangTools.cmake OPTIONAL
  RESULT_VARIABLE CLANG_TOOLS
)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  )
  add_format_target(panda_ros_controllers FILES ${SOURCES} ${HEADERS})
  add_tidy_target(panda_ros_controllers
    FILES ${SOURCES}
    DEPENDS panda_ros_controllers
  )
endif()
