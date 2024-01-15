#ifndef PYRAMIDALKLT_TRACKER_H
#define PYRAMIDALKLT_TRACKER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class Tracker {
private:
    std::vector<Eigen::MatrixXd> I;
    Eigen::Vector2d u;
    int m, wx, wy;
    bool lost;

    static std::vector<Eigen::MatrixXd> makePyramid(const cv::Mat& frame, int m);

    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> partialDerivatives(Eigen::MatrixXd img);

    static Eigen::MatrixXd getBlockInterpolated(
            Eigen::MatrixXd matrix, double startRow, double startCol, int blockRows, int blockCols);

    static std::pair<Eigen::MatrixXd, Eigen::Vector2d>
    calculateGb(Eigen::MatrixXd I, Eigen::MatrixXd J, Eigen::Vector2d gv, Eigen::MatrixXd Ix, Eigen::MatrixXd Iy,
                double top, double left, int rows, int cols);

public:

    void Init(const cv::Mat& frame, cv::Point u, int wx, int wy, int m);

    cv::Point next(const cv::Mat& frame);

    bool isLost() const;
};


#endif //PYRAMIDALKLT_TRACKER_H
