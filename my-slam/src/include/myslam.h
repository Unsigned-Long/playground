#ifndef MYSLAM_H
#define MYSLAM_H

#include "frame.h"

namespace ns_myslam {
  class MySLAM {
  private:

    // the camera and the orb feature
    MonoCamera::Ptr _camera;
    ORBFeature::Ptr _orbFeature;

    // the map points and frames
    std::vector<MapPoint::Ptr> _map;
    std::vector<Frame::Ptr> _frames;
    // {mapPointId, the index in vector "_map"}
    std::unordered_map<int, int> _mptIdxMap;
    // {frameId, the index in vector "_frames"}
    std::unordered_map<int, int> _framesIdxMap;

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
    MySLAM &addFrame(int id, MatPtr grayImg);

  protected:
    /**
     * @brief find good matches from two frames
     *
     * @param lastFrame the first frame
     * @param curFrame the second frame
     * @return std::vector<cv::DMatch> the good matches vector
     */
    std::vector<cv::DMatch> findGoodMatches(Frame::Ptr lastFrame, Frame::Ptr curFrame);

    /**
     * @brief init the scale, run the contrapolar constraint [2d-2d] process
     *
     * @return std::pair<cv::Mat, cv::Mat> the rotation matrix and translate matrix
     */
    std::pair<cv::Mat, cv::Mat> initScale(const std::vector<cv::DMatch> &matches, Frame::Ptr firFrame, Frame::Ptr sedFrame);

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
    MapPoint::Ptr &findMapPoint(int keyPointId);

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
     * @param rotMat the rotation matrix
     * @param transMat the translate matrix
     * @param firFrame the first frame
     * @param sedFrame the second frame
     * @return double the error
     */
    double matchError(const cv::DMatch &match,
                      const Eigen::Matrix3d &rotMat, const Eigen::Vector3d &transMat,
                      Frame::Ptr firFrame, Frame::Ptr sedFrame);

    /**
     * @brief organize the matched key points
     *
     * @param matches the match relationship vector
     * @param firFrame the first frame
     * @param sedFrame the second frame
     * @return std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> {pts1, pts2}
     */
    std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>>
    matchedKeyPoints(const std::vector<cv::DMatch> &matches, Frame::Ptr firFrame, Frame::Ptr sedFrame);
  };
} // namespace ns_myslam

#endif