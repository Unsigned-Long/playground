#include "orbfeature.h"

namespace ns_myslam {
  ORBFeature::ORBFeature() : _feature(cv::ORB::create()) {}

  ORBFeature::Ptr ORBFeature::create() {
    return std::make_shared<ORBFeature>();
  }

  std::vector<cv::KeyPoint> ORBFeature::detect(MatPtr grayImg, const cv::Range &xRange, const cv::Range &yRange) const {
    std::vector<cv::KeyPoint> keyPoints;
    // only detect the win range
    this->_feature->detect((*grayImg)(yRange, xRange), keyPoints);
    return keyPoints;
  }

  std::vector<cv::DMatch> ORBFeature::match(MatPtr img1, std::vector<cv::KeyPoint> &keyPoints1,
                                            MatPtr img2, std::vector<cv::KeyPoint> &keyPoints2) {
    cv::Mat descriptors1, descriptors2;
    // compute the descriptors
    this->_feature->compute(*img1, keyPoints1, descriptors1);
    this->_feature->compute(*img2, keyPoints2, descriptors2);
    // match
    auto matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);
    return matches;
  }

} // namespace ns_myslam
