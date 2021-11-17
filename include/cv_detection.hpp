//
// Created by jzx
//
#ifndef _CV_DETECTION_H_
#define _CV_DETECTION_H_

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/CompressedImage.h"
#include <image_transport/image_transport.h>

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <stdio.h>
#include <cmath>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <sstream>
#include <random>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>


#include"deepsort.hpp"
#include "simple_yolo.hpp"

using cv::Mat;

using namespace cv;

#define CV_GREEN cv::Scalar(0, 255, 0)
#define CV_RED cv::Scalar(0, 0, 255)


class MotionFilter
{
    public:
        MotionFilter()
        {
            location_.left = location_.top = location_.right = location_.bottom = 0;
        }

        void missed()
        {
            init_ = false;
        }

        void update(const DeepSORT::Box &box)
        {

            const float a[] = {box.left, box.top, box.right, box.bottom};
            const float b[] = {location_.left, location_.top, location_.right, location_.bottom};

            if (!init_)
            {
                init_ = true;
                location_ = box;
                return;
            }

            float v[4];
            for (int i = 0; i < 4; ++i)
                v[i] = a[i] * 0.6 + b[i] * 0.4;

            location_.left = v[0];
            location_.top = v[1];
            location_.right = v[2];
            location_.bottom = v[3];
        }

        DeepSORT::Box predict()
        {
            return location_;
        }

    private:
        DeepSORT::Box location_;
        bool init_ = false;
    };




class CvDetection {

    public:
        //ros
        ros::NodeHandle nh_;
        std::shared_ptr<DeepSORT::Tracker> tracker_;
        std::shared_ptr<SimpleYolo::Infer> yolo_;
        std::unordered_map<int, MotionFilter> MotionFilter_;
        void init();
        void imgCallback(const sensor_msgs::CompressedImage::ConstPtr &image_msg);
        void sendMsgs(sensor_msgs::ImagePtr msg);
        void infer(cv::Mat &img);

    private:
        ros::Subscriber img_sub;
        ros::Publisher cvInfo_pub;
        bool debug_ = false;
        bool USE_DEEPSORT_=true;
};

inline void displayDot(cv::Mat &img, const cv::Point2i &dotLoc, double dotScale,
                       const cv::Scalar &color = CV_GREEN) {
    cv::circle(img, dotLoc, dotScale / 2.0, color, dotScale / 2.0);
}

inline void
displayText(cv::Mat &img, std::string &text, const cv::Point2i &textLoc, double fontScale, cv::Scalar color) {
    int fontFace = cv::FONT_HERSHEY_PLAIN;//FONT_HERSHEY_COMPLEX_SMALL;//
    int thickness = MAX(fontScale / 2, 2);
    cv::putText(img, text, textLoc, fontFace, fontScale, std::move(color), thickness, 9);
}

static const char *cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

#endif
