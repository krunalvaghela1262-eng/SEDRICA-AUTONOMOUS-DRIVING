cmake_minimum_required(VERSION 3.8)
project(lidar_merger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(PCL REQUIRED)

add_executable(merger_node src/merger_node.cpp)
target_include_directories(merger_node PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(merger_node ${PCL_LIBRARIES} tf2_ros::tf2_ros tf2_eigen::tf2_eigen)
ament_target_dependencies(merger_node rclcpp sensor_msgs pcl_conversions pcl_ros)

install(TARGETS merger_node DESTINATION lib/${PROJECT_NAME})
ament_package()