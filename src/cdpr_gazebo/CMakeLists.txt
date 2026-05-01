cmake_minimum_required(VERSION 3.0.2)
project(cdpr_gazebo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  gazebo_ros
)
find_package(gazebo REQUIRED)

catkin_package(
  LIBRARIES cdpr_force_plugin
  CATKIN_DEPENDS roscpp std_msgs gazebo_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})
add_definitions(${GAZEBO_CXX_FLAGS})

add_library(cdpr_force_plugin SHARED src/cdpr_force_plugin.cpp)
target_link_libraries(cdpr_force_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

install(TARGETS cdpr_force_plugin
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(DIRECTORY config launch models worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
