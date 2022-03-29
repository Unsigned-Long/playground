#ifndef ORBFEATURE_H
#define ORBFEATURE_H
#include "opencv2/features2d.hpp"

namespace ns_myslam {
  using MatPtr = std::shared_ptr<cv::Mat>;

  class ORBFeature {
  public:
    using Ptr = std::shared_ptr<ORBFeature>;

  private:
    // the FeatureDetector and DescriptorExtractor
    std::shared_ptr<cv::Feature2D> _feature;

  public:
    /**
     * @brief construct a new ORBFeature object
     */
    ORBFeature();

    /**
     * @brief create a shared pointer
     */
    static ORBFeature::Ptr create();

    /**
     * @brief detect the key points
     *
     * @param grayImg the gray image
     * @param xRange the range in x-direction
     * @param yRange the range in y-direction
     * @return std::vector<cv::KeyPoint> the key points vector
     */
    std::vector<cv::KeyPoint> detect(MatPtr grayImg, const cv::Range &xRange, const cv::Range &yRange) const;

    /**
     * @brief match key points
     *
     * @param img1 the first image
     * @param keyPoints1 the key points in first image
     * @param img2 the second iamge
     * @param keyPoints2 the key points in second image
     * @return std::vector<cv::DMatch> the matches
     */
    std::vector<cv::DMatch> match(MatPtr img1, std::vector<cv::KeyPoint> &keyPoints1,
                                  MatPtr img2, std::vector<cv::KeyPoint> &keyPoints2);
  };

} // namespace ns_myslam

#endif