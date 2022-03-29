#ifndef MYSLAM_H
#define MYSLAM_H

#include "frame.h"

namespace ns_myslam {
  class MySLAM {
  private:
    MonoCamera::Ptr _camera;
    ORBFeature::Ptr _orbFeature;

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
    std::pair<cv::Mat, cv::Mat> initScale();

    /**
     * @brief find the frame using the frame id
     *
     * @param frameID the id of the frame
     * @return Frame::Ptr the frame
     */
    Frame::Ptr findFrame(int frameID);

    /**
     * @brief find the map point using key point id
     *
     * @param keyPointId the id of the key point
     * @return MapPoint::Ptr the map point
     */
    MapPoint::Ptr findMapPoint(int keyPointId);
  };
} // namespace ns_myslam

#endif