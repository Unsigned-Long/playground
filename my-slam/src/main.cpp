#include "artwork/timer/timer.h"
#include "myslam.h"
#include "opencv2/opencv.hpp"
#include <filesystem>

/**
 * \brief a function to get all the filenames in the directory
 * \param directory the directory
 * \return the filenames in the directory
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
  auto imgNames = filesInDir("../data/imgs");
  std::sort(imgNames.begin(), imgNames.end());

  std::vector<std::shared_ptr<cv::Mat>> imgs;
  for (const auto &name : imgNames) {
    imgs.push_back(std::make_shared<cv::Mat>(cv::imread(name, cv::IMREAD_GRAYSCALE)));
  }

  float k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;
  float fx = 7.188560000000e+02, fy = 7.188560000000e+02, cx = 6.071928000000e+02, cy = 1.852157000000e+02;

  auto camera = ns_myslam::MonoCamera::create(fx, fy, cx, cy, k1, k2, 0.0f, p1, p2);
  auto orb = ns_myslam::ORBFeature::create();

  ns_myslam::MySLAM slam(camera, orb);

  for (const auto &img : imgs) {
    slam.addFrame(img);
    ns_log::process("current pose: ", slam.currentPose().translation().transpose());
    std::cout << "\n";
  }
  return 0;
}
