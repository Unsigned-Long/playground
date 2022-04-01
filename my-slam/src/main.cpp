#include "artwork/timer/timer.h"
#include "myslam.h"
#include "opencv2/opencv.hpp"
#include <filesystem>

/**
 * @brief a function to get all the filenames in the directory
 * @param directory the directory
 * @return the filenames in the directory
 */
std::vector<std::string> filesInDir(const std::string &directory) {
  std::vector<std::string> files;
  for (const auto &elem : std::filesystem::directory_iterator(directory))
    if (elem.status().type() != std::filesystem::file_type::directory)
      files.push_back(std::filesystem::canonical(elem.path()).c_str());
  return files;
}

int main(int argc, char const *argv[]) {
  // read data
  auto imgNames = filesInDir("/home/csl/kitti/sequence/image_0");
  std::sort(imgNames.begin(), imgNames.end());

  double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
  auto camera = ns_myslam::MonoCamera::create(fx, fy, cx, cy);
  auto orb = ns_myslam::ORBFeature::create();
  ns_myslam::MySLAM slam(camera, orb);

  for (const auto &name : imgNames) {
    slam.addFrame(std::make_shared<cv::Mat>(cv::imread(name, cv::IMREAD_GRAYSCALE)));
  }
  return 0;
}
