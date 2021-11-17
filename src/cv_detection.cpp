#include "cv_detection.hpp"



using namespace std;
using namespace cv;

static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
{
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i)
    {
    case 0:
        r = v;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = v;
        b = p;
        break;
    case 2:
        r = p;
        g = v;
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = v;
        break;
    case 4:
        r = t;
        g = p;
        b = v;
        break;
    case 5:
        r = v;
        g = p;
        b = q;
        break;
    default:
        r = 1;
        g = 1;
        b = 1;
        break;
    }
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    ;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

static bool exists(const string &path)
{

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

void CvDetection::init()
{
    ROS_INFO("<< cv detection go!");
    img_sub = nh_.subscribe<sensor_msgs::CompressedImage>("camera/image_raw/compressed", 1,
                                                          &CvDetection::imgCallback, this);
    //cvInfo_pub = nh_.advertise<wa_ros_msgs::CvInfo>("/cv/cv_info", 1);
    nh_.param<bool>("is_debug", debug_, true);

    //0.load model build yolo class
    printf("TRTVersion: %s\n", SimpleYolo::trt_version());
    int device_id = 0;
    // string model = "yolox_s";
    auto type = SimpleYolo::Type::V5;
    auto mode = SimpleYolo::Mode::FP32;
    string model_path = "/home/jzx/IMPORTANT_MODELS/detection_model.trt";
    SimpleYolo::set_device(device_id);
    float confidence_threshold = 0.4f;
    float nms_threshold = 0.5f;
    yolo_ = SimpleYolo::create_infer(model_path, type, device_id, confidence_threshold, nms_threshold);
    if (yolo_ == nullptr)
    {
        printf("Yolo is nullptr\n");
        return;
    }
    //1.初始化追踪器
    if (USE_DEEPSORT_)
    {
        auto config = DeepSORT::TrackerConfig();
        config.has_feature = false;
        config.max_age = 150;
        config.nbuckets = 150;
        config.distance_threshold = 100.0f;
        config.set_per_frame_motion({0.05, 0.02, 0.1, 0.02,
                                     0.08, 0.02, 0.1, 0.02});
        tracker_ = DeepSORT::create_tracker(config);
    }
}


void CvDetection::infer(cv::Mat &img)
{
    auto det_objs = yolo_->commit(img).get();
    cout << "det objets size : " << to_string(det_objs.size()) << std::endl;
    
    if (USE_DEEPSORT_)
    {
        vector<DeepSORT::Box> boxes;
        for (int i = 0; i < det_objs.size(); ++i)
        {
            auto &det_obj = det_objs[i];
            //std::cout<<to_string(det_obj.class_label)<<std::endl;
            if (det_obj.class_label == 2)
            { //只有在检测是car的时候才更新追踪
                auto track_box = DeepSORT::convert_to_box(det_obj);
                //track_box.feature = det_obj.feature;
                boxes.emplace_back(std::move(track_box));
            }
        }
        //debug
        // show_result(img, det_objs);
        tracker_->update(boxes);
        auto final_objects = tracker_->get_objects();
        for (int i = 0; i < final_objects.size(); ++i)
        {
            std::cout << to_string(i) << std::endl;
            auto &obj = final_objects[i];
            auto &filter = MotionFilter_[obj->id()];
            if (obj->time_since_update() == 0 && obj->state() == DeepSORT::State::Confirmed)
            {
                uint8_t b, g, r;
                tie(b, g, r) = random_color(obj->id());
                auto loaction = obj->last_position();
                filter.update(loaction);
                loaction = filter.predict();
                cv::rectangle(img, cv::Point(loaction.left, loaction.top), cv::Point(loaction.right, loaction.bottom), cv::Scalar(b, g, r), 5);
                auto name = cocolabels[0]; //loaction.class_label
                auto caption = cv::format("%s %.2f", name, loaction.confidence);
                auto id = obj->id();
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(img, cv::Point(loaction.left - 3, loaction.top - 33), cv::Point(loaction.left + width, loaction.top), cv::Scalar(b, g, r), -1);
                //cv::putText(image, caption, cv::Point(loaction.left, loaction.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
                cv::putText(img, to_string(id), cv::Point(loaction.left, loaction.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            }
            else
            {
                filter.missed();
            }
        }

        cv::imshow("inference_by_yolov5+deepsort", img);
        cv::waitKey(1);
    }
    else
    {
        for (auto &obj : det_objs)
        {
            if(obj.class_label == 2){
                uint8_t b, g, r;
                tie(b, g, r) = random_color(obj.class_label);
                cv::rectangle(img, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

                auto name = cocolabels[obj.class_label];
                auto caption = cv::format("%s %.2f", name, obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(img, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                cv::putText(img, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            }
        }
        cv::imshow("debug", img);
        cv::waitKey(1);
    }
}

void CvDetection::imgCallback(const sensor_msgs::CompressedImage::ConstPtr &image_msg)
{
    try
    {
        if (image_msg->header.seq % 2 == 0)
        {
            cv::Mat image = cv::imdecode(cv::Mat(image_msg->data), 1); //convert compressed image data to cv::Mat
            infer(image);
        }
    }
    catch (cv_bridge::Exception &e)
    {
        std::cout << "could not " << std::endl;
    }
}

void CvDetection::sendMsgs(sensor_msgs::ImagePtr msg)
{

    std::cout << "publish the cv result" << std::endl;
    cvInfo_pub.publish(msg);
}
