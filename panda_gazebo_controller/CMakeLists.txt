cmake_minimum_required(VERSION 3.4)
project(panda_gazebo_controller)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED
  COMPONENTS
  controller_interface
  dynamic_reconfigure
  control_toolbox
  eigen_conversions
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
find_package(Franka 0.7.0 REQUIRED)
find_package(Boost REQUIRED system filesystem date_time thread)

generate_messages()

generate_dynamic_reconfigure_options(
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
  LIBRARIES panda_gazebo_controller
  CATKIN_DEPENDS
    controller_interface
    controller_interface
    dynamic_reconfigure
    eigen_conversions
    franka_hw
    geometry_msgs
    hardware_interface
    message_runtime
    pluginlib
    realtime_tools
    roscpp
  DEPENDS Franka
)


# add_dependencies(panda_gazebo_controller
#   ${${PROJECT_NAME}_EXPORTED_TARGETS}
#   ${catkin_EXPORTED_TARGETS}
#   ${PROJECT_NAME}_generate_messages_cpp
#   ${PROJECT_NAME}_gencpp
#   ${PROJECT_NAME}_gencfg
# )

include_directories(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)


# target_link_libraries(panda_gazebo_controller PUBLIC
#   ${Franka_LIBRARIES}
#   ${catkin_LIBRARIES}
# )
#
# target_include_directories(panda_gazebo_controller SYSTEM PUBLIC
#   ${Franka_INCLUDE_DIRS}
#   ${EIGEN3_INCLUDE_DIRS}
#   ${catkin_INCLUDE_DIRS}
# )
# target_include_directories(panda_gazebo_controller PUBLIC
#   include
# )


## Installation
# install(TARGETS panda_gazebo_controller
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#   )

###########
## Build ##
###########

include_directories(${THIS_PACKAGE_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})

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
  add_format_target(panda_gazebo_controller FILES ${SOURCES} ${HEADERS})
  add_tidy_target(panda_gazebo_controller
    FILES ${SOURCES}
    DEPENDS panda_gazebo_controller
  )
endif()
