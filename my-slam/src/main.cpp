#include "artwork/timer/timer.h"
#include "myslam.h"
#include "opencv2/opencv.hpp"

int main(int argc, char const *argv[]) {
  auto img1 = std::make_shared<cv::Mat>(cv::imread("../img/1.png", cv::IMREAD_GRAYSCALE));
  auto img2 = std::make_shared<cv::Mat>(cv::imread("../img/2.png", cv::IMREAD_GRAYSCALE));

  double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;
  double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;

  auto camera = ns_myslam::MonoCamera::create(fx, fy, cx, cy, k1, k2, 0.0f, p1, p2);
  auto orb = ns_myslam::ORBFeature::create();

  ns_myslam::MySLAM slam(camera, orb);
  ns_timer::Timer<> timer;
  timer.reStart();
  slam.addFrame(0, img1);
  std::cout << timer.last_elapsed("fir frame") << std::endl;
  timer.reStart();
  slam.addFrame(1, img2);
  std::cout << timer.last_elapsed("sed frame") << std::endl;
  return 0;
}