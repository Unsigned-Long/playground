#include "camera.h"
#include "opencv2/opencv.hpp"

int main(int argc, char const *argv[]) {
  auto img = std::make_shared<cv::Mat>(cv::imread("../img/distorted.png", cv::IMREAD_GRAYSCALE));

  double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
  double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

  ns_myslam::MonoCamera camera(fx, fy, cx, cy, k1, k2, 0.0f, p1, p2);
  auto [xRange, yRange] = camera.undistort(img);

  std::cout << "xRange: " << xRange << " yRange: " << yRange << std::endl;
  std::cout << camera << std::endl;

  cv::imshow("win", *img);
  cv::waitKey(0);
  return 0;
}
