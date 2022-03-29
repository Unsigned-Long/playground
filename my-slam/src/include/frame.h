#ifndef FRAME_H
#define FRAME_H

#include "camera.h"
#include "mappoint.h"
#include "opencv2/core.hpp"
#include "orbfeature.h"
#include <memory>

namespace ns_myslam {
  class MySLAM;
  class Frame {
  public:
    using Ptr = std::shared_ptr<Frame>;
    friend class MySLAM;

  private:
    // the id of the frame, used in the map point
    const int _id;
    // the gray image
    MatPtr _grayImg;

    // the key points in this frame
    std::vector<cv::KeyPoint> _keyPoints;
    // map point corresponding to key point, existed only when the key point is matched width the next image
    std::unordered_map<int, MapPoint::Ptr> _mapPoints;

  public:
    /**
     * @brief construct a new frame
     *
     * @param id the id of this frame
     * @param grayImg the gray image
     * @param camera the camera
     * @param orbFeature the orb feature
     */
    Frame(int id, MatPtr grayImg, MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature);

    /**
     * @brief create a frame [shared ptr]
     */
    static Frame::Ptr create(int id, MatPtr grayImg, MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature);

  protected:
    /**
     * @brief undistort the image and detect the key points in win range
     *
     * @param camera the camera
     * @param orbFeature the orb feature
     */
    void pretreatment(MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature);
  };

} // namespace ns_myslam

#endif