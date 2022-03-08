#include "harris.h"

namespace ns_harris
{

#pragma region static class Harris

    std::vector<cv::Point> Harris::cornerDetector(const cv::Mat &grayImg, cv::Mat &dst, int blockSize, float alpha, float threshold, const Output out)
    {
        auto rows = grayImg.rows;
        auto cols = grayImg.cols;
        // three channels : [Ix, Iy, R]
        cv::Mat gradImg(rows, cols, CV_32FC3);
        // calculate the grad using Sobel operator
        for (int i = 1; i != rows - 1; ++i)
        {
            auto gradPtr = gradImg.ptr<_Float32>(i);
            auto grayPtr = grayImg.ptr<uchar>(i);
            for (int j = 1; j != cols - 1; ++j)
            {
                auto l1 = grayPtr[j - 1 - cols];
                auto l2 = grayPtr[j - 1];
                auto l3 = grayPtr[j - 1 + cols];
                auto r1 = grayPtr[j + 1 - cols];
                auto r2 = grayPtr[j + 1];
                auto r3 = grayPtr[j + 1 + cols];
                auto m1 = grayPtr[j - cols];
                auto m2 = grayPtr[j + cols];
                // Ix
                gradPtr[j * 3 + 0] = -1 * l1 - 2 * l2 - 1 * l3 + r1 + 2 * r2 + r3;
                // Iy
                gradPtr[j * 3 + 1] = l1 + 2 * m1 + r1 - l3 - 2 * m2 - r3;
            }
        }
#pragma region display
        float min = FLT_MAX;
        float max = 0.0;
#pragma endregion
        auto edge = (blockSize - 1) / 2;
        for (int i = edge + 1; i != rows - edge - 1; ++i)
        {
            auto gradPtr = gradImg.ptr<_Float32>(i);
            for (int j = edge + 1; j != cols - edge - 1; ++j)
            {
                // calculate the M matrix
                Eigen::Matrix2f tempM = Eigen::Matrix2f::Zero();
                for (int k = i - edge; k != i + edge + 1; ++k)
                {
                    for (int l = j - edge; l != j + edge + 1; ++l)
                    {
                        auto dis = 3 * ((i - k) * cols + (j - l));
                        auto tempPtr = &(gradPtr[j * 3 + 0]) - dis;
                        auto Ix = tempPtr[0];
                        auto Iy = tempPtr[1];
                        float x = std::sqrt((i - k) * (i - k) + (j - l) * (j - l));
                        auto w = Gaussian<float>::gaussianNormalized(x, 0, 1);
                        tempM(0, 0) += w * Ix * Ix;
                        tempM(0, 1) += w * Ix * Iy;
                        tempM(1, 0) += w * Ix * Iy;
                        tempM(1, 1) += w * Iy * Iy;
                    }
                }
                // calculate the R value
                auto R = tempM.determinant() - alpha * std::pow(tempM.trace(), 2);
                gradPtr[j * 3 + 2] = R;
#pragma region display
                if (min > R)
                    min = R;
                if (max < R)
                    max = R;
#pragma endregion display
            }
        }
#pragma region display
        dst = cv::Mat(rows, cols, CV_8UC3);
        std::vector<cv::Point> pos;
        auto range = max - min;
        for (int i = 0; i != rows; ++i)
        {
            auto dstPtr = dst.ptr<uchar>(i);
            auto grayPtr = grayImg.ptr<uchar>(i);
            auto gradPtr = gradImg.ptr<_Float32>(i);
            for (int j = 0; j != cols; ++j)
            {
                if (gradPtr[j * 3 + 2] > threshold)
                    pos.push_back(cv::Point2i(j, i));
                /**
                 * draw circles in the dst image
                 */
                if (out == Output::MARK)
                    dstPtr[j * 3 + 0] = dstPtr[j * 3 + 1] = dstPtr[j * 3 + 2] = grayPtr[j];
                else
                {
                    /**
                 * output the R intresting image
                 */
                    auto val = (gradPtr[j * 3 + 2] - min) / range * 255;
                    dstPtr[j * 3 + 0] = dstPtr[j * 3 + 1] = dstPtr[j * 3 + 2] = val;
                }
            }
        }
        if (out == Output::MARK)
            for (const auto &point : pos)
                cv::circle(dst, point, 10, cv::Scalar(0, 0, 255), 2);
#pragma endregion
        return pos;
    }
#pragma endregion

} // namespace ns_test