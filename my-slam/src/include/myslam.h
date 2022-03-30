#ifndef MYSLAM_H
#define MYSLAM_H

#include "frame.h"

namespace ns_myslam {
  class MySLAM {
  private:
    // the camera and the orb feature
    MonoCamera::Ptr _camera;
    ORBFeature::Ptr _orbFeature;

    // the map points and frames, and the index is the id
    std::vector<MapPoint::Ptr> _map;
    std::vector<Frame::Ptr> _frames;

    bool _initSystem;
    bool _initScale;

  public:
    MySLAM(MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature);

    /**
     * @brief add a new frame to the slam system
     *
     * @param id the id of the new frame
     * @param grayImg the gray image
     */
    MySLAM &addFrame(MatPtr grayImg);

  protected:
    /**
     * @brief find good matches from two frames
     *
     * @param frame1 the first frame
     * @param frame2 the second frame
     * @return std::vector<cv::DMatch> the good matches vector
     */
    std::vector<cv::DMatch> findGoodMatches(Frame::Ptr frame1, Frame::Ptr frame2);

    /**
     * @brief init the scale, run the contrapolar constraint [2d-2d] process
     *
     * @param matches the matches
     * @param frame1 the first frame
     * @param frame2 the second frame
     * @return std::pair<cv::Mat, cv::Mat> the rotation matrix and translate matrix
     */
    std::pair<cv::Mat, cv::Mat> initScale(const std::vector<cv::DMatch> &matches, Frame::Ptr frame1, Frame::Ptr frame2);

    /**
     * @brief find the frame using the frame id
     *
     * @param frameID the id of the frame
     * @return Frame::Ptr the frame
     */
    Frame::Ptr &findFrame(int frameID);

    /**
     * @brief find the map point using key point id
     *
     * @param keyPointId the id of the key point
     * @return MapPoint::Ptr the map point
     */
    MapPoint::Ptr &findMapPoint(int mptId);

    /**
     * @brief transfrom a vector to an antisymmetric matrix
     *
     * @param vec the vector[3d]
     * @return Eigen::Matrix3d
     */
    Eigen::Matrix3d vec2AntiMat(const Eigen::Vector3d &vec);

    /**
     * @brief calculate the match error
     *
     * @param match the match relationship
     * @param rotMat_21 the rotation matrix from frame1 to frame2
     * @param transMat_21 the translate matrix from frame1 to frame2
     * @param frame2 the first frame
     * @param frame1 the second frame
     * @return double the error
     */
    double matchError(const cv::DMatch &match,
                      const Eigen::Matrix3d &rotMat_21, const Eigen::Vector3d &transMat_21,
                      Frame::Ptr frame1, Frame::Ptr frame2);

    /**
     * @brief organize the matched key points
     *
     * @param matches the match relationship vector
     * @param frame1 the first frame
     * @param frame2 the second frame
     * @return std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> {pts1, pts2}
     */
    std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>>
    matchedKeyPoints(const std::vector<cv::DMatch> &matches, Frame::Ptr frame1, Frame::Ptr frame2);

    Eigen::Vector3d triangulation(const Eigen::Vector2d &pixel1, const Eigen::Vector2d &pixel2,
                                  const Eigen::Matrix3d &rotMat_21, const Eigen::Vector3d &transMat_21);

    cv::KeyPoint &findKeyPoint(int frameId, int kptIdx);
  };
} // namespace ns_myslam

#endif