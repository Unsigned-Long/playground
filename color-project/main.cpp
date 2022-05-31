#include "colorPrj.h"
#include <algorithm>
#include <fstream>
#include <list>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <string>
#include <vector>

std::vector<std::string> split(const std::string &str) {
  std::vector<std::string> vec;
  auto iter = str.cbegin();
  while (true) {
    auto pos = std::find(iter, str.cend(), ',');
    vec.push_back(std::string(iter, pos));
    if (pos == str.cend())
      break;
    iter = ++pos;
  }
  return vec;
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./colorProject dataFilePath" << std::endl;
    return 1;
  }
  using namespace ns_clp;
  std::fstream file(argv[1], std::ios::in);
  std::string strLine;
  std::list<cv::Point3f> ps;
  while (std::getline(file, strLine)) {
    auto vec = split(strLine);
    ps.push_back(cv::Point3f(
        std::stod(vec[0]),
        std::stod(vec[1]),
        std::stod(vec[3])));
  }
  file.close();
  auto min_max = std::minmax_element(ps.cbegin(), ps.cend(), [](const cv::Point3f &p1, const cv::Point3f &p2) { return p1.z < p2.z; });
  auto min = min_max.first->z;
  auto max = min_max.second->z;
  cv::Mat img(700, 500, CV_8UC3);
  for (int i = 0; i != img.rows; ++i) {
    auto ptr = img.ptr<uchar>(i);
    for (int j = 0; j != img.cols; ++j) {
      auto [r, g, b] = project(ps.front().z, min, max, false, 0, Color::panchromatic);
      ptr[j * img.channels() + 0] = b;
      ptr[j * img.channels() + 1] = g;
      ptr[j * img.channels() + 2] = r;
      ps.pop_front();
    }
  }
  cv::imshow("win", img);
  cv::waitKey(0);
  return 0;
}
