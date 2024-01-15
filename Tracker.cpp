#include "Tracker.h"

// external 1px border in every layer
std::vector<Eigen::MatrixXd> Tracker::makePyramid(const cv::Mat& frame, int m) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    std::vector<Eigen::MatrixXd> pyramid{};
    pyramid.emplace_back(gray.rows + 2, gray.cols + 2);
    for (int y = 0; y < gray.rows; y++)
        for (int x = 0; x < gray.cols; x++)
            pyramid[0](y + 1, x + 1) = gray.at<uchar>(y, x) / 255.0;
    pyramid[0].row(0) = pyramid[0].row(1);
    pyramid[0].row(pyramid[0].rows() - 1) = pyramid[0].row(pyramid[0].rows() - 2);
    pyramid[0].col(0) = pyramid[0].col(1);
    pyramid[0].col(pyramid[0].cols() - 1) = pyramid[0].col(pyramid[0].cols() - 2);
    Eigen::Matrix3d filter = Eigen::Vector3d(0.25, 0.5, 0.25)
                             * Eigen::Vector3d(0.25, 0.5, 0.25).transpose();
    for (int l = 1; l < m; ++l) {
        pyramid.emplace_back(pyramid[l - 1].rows() / 2 + 1, pyramid[l - 1].cols() / 2 + 1);
        for (int y = 0; y < pyramid[l].rows() - 2; ++y) {
            for (int x = 0; x < pyramid[l].cols() - 2; ++x) {
                Eigen::MatrixXd neighborhood =
                        pyramid[l - 1].block(2 * y, 2 * x, 3, 3);
                pyramid[l](y + 1, x + 1) = (neighborhood.array() * filter.array()).sum();
            }
        }
        pyramid[l].row(0) = pyramid[l].row(1);
        pyramid[l].row(pyramid[l].rows() - 1) = pyramid[l].row(pyramid[l].rows() - 2);
        pyramid[l].col(0) = pyramid[l].col(1);
        pyramid[l].col(pyramid[l].cols() - 1) = pyramid[l].col(pyramid[l].cols() - 2);
    }
    return pyramid;
}

// returns image with -2 size both axis
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Tracker::partialDerivatives(Eigen::MatrixXd img) {
    Eigen::MatrixXd resultX(img.rows() - 2, img.cols() - 2);
    Eigen::MatrixXd resultY(img.rows() - 2, img.cols() - 2);
    for (int y = 0; y < img.rows() - 2; ++y) {
        for (int x = 0; x < img.cols() - 2; ++x) {
            resultX(y, x) = (img(y + 1, x + 2) - img(y + 1, x)) / 2.0;
            resultY(y, x) = (img(y + 2, x + 1) - img(y, x + 1)) / 2.0;
        }
    }
    return {resultX, resultY};
}

Eigen::MatrixXd
Tracker::getBlockInterpolated(Eigen::MatrixXd matrix, double startRow, double startCol, int blockRows, int blockCols) {
    // tl, tr, bl, br
    Eigen::MatrixXd tlblock = matrix.block(
            (int)floor(startRow),
            (int)floor(startCol),
            blockRows, blockCols);
    Eigen::MatrixXd trblock = matrix.block(
            (int)floor(startRow),
            (int)ceil(startCol),
            blockRows, blockCols);
    Eigen::MatrixXd blblock = matrix.block(
            (int)ceil(startRow),
            (int)floor(startCol),
            blockRows, blockCols);
    Eigen::MatrixXd brblock = matrix.block(
            (int)ceil(startRow),
            (int)ceil(startCol),
            blockRows, blockCols);
    double ax = fmod(startCol, 1.0), ay = fmod(startRow, 1.0);
    Eigen::MatrixXd result =
            (1 - ax) * (1 - ay) * tlblock
            + ax * (1 - ay) * trblock
            + (1 - ax) * ay * blblock
            + ax * ay * brblock;
    return result;
}

std::pair<Eigen::MatrixXd, Eigen::Vector2d>
Tracker::calculateGb(Eigen::MatrixXd I, Eigen::MatrixXd J, Eigen::Vector2d gv, Eigen::MatrixXd Ix, Eigen::MatrixXd Iy,
                     double top, double left, int rows, int cols) {
    Eigen::MatrixXd Iblock = getBlockInterpolated(
            std::move(I),
            top,left,
            rows,cols);
    Eigen::MatrixXd Jblock = getBlockInterpolated(
            std::move(J),
            top + gv.y(),
            left + gv.x(),
            rows,cols);
    Eigen::MatrixXd dIblock = Iblock - Jblock;
    Eigen::MatrixXd Ixblock = getBlockInterpolated(
            std::move(Ix),
            top,left,
            rows,cols);
    Eigen::MatrixXd Iyblock = getBlockInterpolated(
            std::move(Iy),
            top,left,
            rows,cols);
    double dIx = (dIblock.array() * Ixblock.array()).sum();
    double dIy = (dIblock.array() * Iyblock.array()).sum();
    Eigen::Vector2d b(dIx, dIy);

    double Ix2 = (Ixblock.array() * Ixblock.array()).sum();
    double Ixy = (Ixblock.array() * Iyblock.array()).sum();
    double Iy2 = (Iyblock.array() * Iyblock.array()).sum();
    Eigen::Matrix2d G;
    G << Ix2, Ixy,
         Ixy, Iy2;

    return {G, b};
}

void Tracker::Init(const cv::Mat& frame, cv::Point u, int wx, int wy, int m) {
    this->I = makePyramid(frame, m);
    this->u = Eigen::Vector2d(u.x, u.y);
    this->wx = wx;
    this->wy = wy;
    this->m = m;
}

cv::Point Tracker::next(const cv::Mat& frame) {
    std::vector<Eigen::MatrixXd> J = makePyramid(frame, m);
    Eigen::Vector2d g(0, 0);
    for (int l = m - 1; l >= 0; --l) {
        Eigen::Vector2d ul = u / (1 << l);
        auto [Ix, Iy] = partialDerivatives(I[l]);
        Eigen::MatrixXd Iblock = I[l].block(1, 1, I[l].rows() - 2, I[l].cols() - 2);
        Eigen::MatrixXd Jblock = J[l].block(1, 1, J[l].rows() - 2, J[l].cols() - 2);
        int Sx = (int)Iblock.cols() - 1, Sy = (int)Iblock.rows() - 1;
        Eigen::Vector2d v(0, 0), nu(0, 1);
        int k = 0;
        while (nu.norm() > 0.03) {
            double top = ul.y() - wy;
            double left = ul.x() - wx;
            int rows = wy * 2 + 1;
            int cols = wx * 2 + 1;

            while (top < 0 || top + g.y() + v.y() < 0) {
                top += 1;
                rows -= 1;
            }
            while (left < 0 || left + g.x() + v.x() < 0) {
                left += 1;
                cols -= 1;
            }
            while (top + rows > Sy || top + rows + g.y() + v.y() > Sy)
                rows -= 1;
            while (left + cols > Sx || left + cols + g.x() + v.x() > Sx)
                cols -= 1;

            if (rows <= 0 || cols <= 0 || ++k > 50) {
                lost = true;
                return {0, 0};
            }

            auto [G, b] = calculateGb(
                    Iblock, Jblock,
                    g + v, Ix, Iy, top, left, rows, cols);

            nu = G.inverse() * b;
            v += nu;
        }
        g = 2 * (g + v);
    }
    u += g / 2;
    I = J;
    lost = false;
    return {(int)u.x(), (int)u.y()};
}

bool Tracker::isLost() const {
    return lost;
}
