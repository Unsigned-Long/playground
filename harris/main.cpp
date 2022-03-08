#include "harris.h"

void foo()
{
    auto img = cv::imread("../images/img4.png", cv::IMREAD_UNCHANGED);
    cv::Mat dst;
    // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    auto pos = ns_harris::Harris::cornerDetector(img, dst, 3, 0.05, 1E11, ns_harris::Harris::Output::RVALUE);
    cv::namedWindow("win", cv::WINDOW_FREERATIO);
    cv::imshow("win", dst);
    cv::waitKey(0);
    cv::imwrite("../images/result_rvalue3.png", dst);
    for (const auto &elem : pos)
        std::cout << elem << std::endl;
}

int main(int argc, char const *argv[])
{
    ::foo();
    return 0;
}
