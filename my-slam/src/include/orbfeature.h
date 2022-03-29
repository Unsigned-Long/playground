#ifndef ORBFEATURE_H
#define ORBFEATURE_H
#include "opencv2/features2d.hpp"

namespace ns_myslam {
  using MatPtr = std::shared_ptr<cv::Mat>;

  class ORBFeature {
  public:
    using Ptr = std::shared_ptr<ORBFeature>;

  private:
    std::shared_ptr<cv::FeatureDetector> _detector;

  public:
    ORBFeature();

    static ORBFeature::Ptr create();

    std::vector<cv::KeyPoint> detect(MatPtr grayImg) const;

    std::vector<cv::DMatch> match(MatPtr img1, std::vector<cv::KeyPoint> &keyPoints1,
                                  MatPtr img2, std::vector<cv::KeyPoint> &keyPoints2);
  };

} // namespace ns_myslam

#endif