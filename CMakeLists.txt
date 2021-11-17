cmake_minimum_required(VERSION 2.8.3)
project(cv_detection)
add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")
#set(OpenCV_DIR   "/data/datav/expstation/lean/opencv4.2.0/lib/cmake/opencv4/")
set(CUDA_DIR     "/usr/local/cuda-11.3")
set(CUDNN_DIR    "/usr/local/cuda-11.3/include")
set(TENSORRT_DIR "/usr/src/tensorrt")




#指定依赖包，生成对应的环境变量
find_package(catkin REQUIRED
        roscpp
        rospy
        geometry_msgs
        sensor_msgs
        std_msgs
        dynamic_reconfigure)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
#catkin宏命令，用于配置ROS的package配置文件和cmake文件，这个命令必须在add_library()和add_executable()之前调用，
# 有五个可选参数
# 1.INCLUDE_DIRS 导出包的includ路径
# 2.LIBRARIES-导出项目中的库
# 3.CATKIN_DEPENDS 该项目依赖的其他catkin项目
# 4.DEPENDS -该项目所依赖的非catkin cmake项目
# 5.CFG_EXTRAS  -其他配置选项
catkin_package(
        INCLUDE_DIRS include
        LIBRARIES cv_detection
        CATKIN_DEPENDS roscpp rospy geometry_msgs sensor_msgs std_msgs dynamic_reconfigure
        #  DEPENDS system_lib
)


#指定c++的头文件路径
include_directories(
        include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_DIR}/include
        ${TENSORRT_DIR}/include
        ${CUDNN_DIR}/include
)

#指定链接库的路径
link_directories(
    ${catkin_LIBRARY_DIRS}
    ${TENSORRT_DIR}/lib
    ${CUDA_DIR}/lib64
    ${CUDNN_DIR}/lib
)

#添加自定义消息
# add_message_files()

#添加自定义服务
# add_service_files()

#生成消息服务
#generate_messages()



# 指定SRC环境变量
set(SRC
        src/main.cpp
        src/cv_detection.cpp
        src/deepsort.cpp
        src/simple_yolo.cu
        )
# 指定可执行文件 有cuda文件和无cuda文件
# add_executable(${PROJECT_NAME} ${SRC})
cuda_add_executable(${PROJECT_NAME} ${SRC})
# 指定链接可执行文件
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${catkin_LIBRARIES})

target_include_directories(
    ${PROJECT_NAME}  PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

target_link_libraries(
    ${PROJECT_NAME}  -lnvinfer -lnvinfer_plugin -lnvparsers -lnvonnxparser
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    )


# 指定可执行依赖
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})


# 安装目标文件到本地系统
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
