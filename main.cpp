#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "Tracker.h"

namespace fs = std::filesystem;

int main() {
    std::string source_folder_path = "./source_frames";
    std::string output_folder_path = "./output_frames";
    if (!fs::exists(source_folder_path) || !fs::is_directory(source_folder_path)) {
        std::cerr << "Folder is not exist or it's not a directory" << std::endl;
        return 1;
    }

    // load filenames
    std::vector<std::string> filenames;
    for (const auto& entry : fs::directory_iterator(source_folder_path)) {
        if (entry.is_regular_file()) {
            filenames.emplace_back(entry.path().filename().string());
        }
    }

    // load first frame and initialize Tracker
    cv::Mat frame = cv::imread(source_folder_path + '/' + filenames[0]);
    if (frame.empty()) {
        std::cerr << "Can't load frame" << std::endl;
        return 1;
    }
    cv::Point object_location(298, 92);
    Tracker tracker;
    tracker.Init(frame, object_location, 5, 5, 3);
    cv::circle(frame, object_location, 10, cv::Scalar(0, 0, 255), 2);
    cv::imwrite(output_folder_path + '/' + filenames[0], frame);

    // iterate through other frames
    for (int i = 1; i < 75; ++i) {
        frame = cv::imread(source_folder_path + '/' + filenames[i]);
        if (frame.empty()) {
            std::cerr << "Can't load frame" << std::endl;
            return 1;
        }
        object_location = tracker.next(frame);
        cv::circle(frame, object_location, 10, cv::Scalar(0, 0, 255), 2);
        cv::imwrite(output_folder_path + '/' + filenames[i], frame);
        std::cout << filenames[i] << std::endl;
    }

    return 0;
}
