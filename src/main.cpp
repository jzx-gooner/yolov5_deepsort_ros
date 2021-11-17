//
// Created by jzx
//
#include "cv_detection.hpp"
int main(int argc, char **argv) {
    ros::init(argc, argv, "cv_detection");
    CvDetection cd;
    cd.init();
    ROS_INFO("<< cv segmentation go!");
    ros::spin();
    return 0;
}