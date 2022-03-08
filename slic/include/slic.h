#pragma once
#include <iostream>
#include <string>
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/highgui.hpp"

namespace ns_slic
{
    class SLIC
    {
    public:
    private:
        SLIC() = delete;

    public:
        /**
         * \brief using slic algorithm to deal with image
         * \param src the source image
         * \param K the class number
         * \param pixelize whether to pixelize the image
         * \param drawCens whether to draw the centers on the image
         * \param iterCount max iteration count
         * \return the result image
         */
        static cv::Mat process(const cv::Mat &src, std::size_t K,
                               bool pixelize = false,
                               bool drawCens = false,
                               int iterCount = 10);
    };
} // namespace ns_slic
