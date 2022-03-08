#pragma once

#include <iostream>
#include <math.h>
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "eigen3/Eigen/Dense"

namespace ns_harris
{
#pragma region static class Gaussian
    template <typename _Ty>
    class Gaussian
    {
    public:
        using value_type = _Ty;

    private:
        Gaussian() = default;

    public:
        // the max value is sigma * std::sqrt(2.0 * M_PI)
        static value_type gaussian(value_type x, value_type mean, value_type sigma);
        // the max value is 1
        static value_type gaussianNormalized(value_type x, value_type mean, value_type sigma);
    };

    template <typename _Ty>
    typename Gaussian<_Ty>::value_type Gaussian<_Ty>::gaussian(value_type x, value_type mean, value_type sigma)
    {
        return Gaussian<_Ty>::gaussianNormalized(x, mean, sigma) / (sigma * std::sqrt(2.0 * M_PI));
    }

    template <typename _Ty>
    typename Gaussian<_Ty>::value_type Gaussian<_Ty>::gaussianNormalized(value_type x, value_type mean, value_type sigma)
    {
        return std::pow(M_E, -0.5 * (std::pow((x - mean) / sigma, 2)));
    }
#pragma endregion

#pragma region static class Harris
    class Harris
    {
    public:
        // type of the cornerDetector's output image
        enum class Output
        {
            RVALUE,
            MARK
        };

    private:
        Harris() = default;

    public:
        /**
         * \brief the main function
         * \param garyImg the gray image to detect
         * \param dst the output image of the function
         * \param blockSize the size of the calculateing block
         * \param alpha the value to control the R value
         * \param threshold the threshold of the R value
         * \param out the output image's type
         * \return the corner points' position in the grayImg
         */
        static std::vector<cv::Point> cornerDetector(const cv::Mat &grayImg, cv::Mat &dst, int blockSize, float alpha, float threshold, const Output out = Output::MARK);
    };
#pragma endregion
} // namespace ns_test
