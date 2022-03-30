#ifndef FRAME_H
#define FRAME_H

#include "artwork/logger/logger.h"
#include "artwork/timer/timer.h"
#include "camera.h"
#include "mappoint.h"
#include "opencv2/core.hpp"
#include "orbfeature.h"
#include "sophus/se3.hpp"
#include <memory>

namespace ns_myslam {

  static ns_timer::Timer<> timer = ns_timer::Timer<>();

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
    std::vector<cv::KeyPoint> _kpts;

    // the related map point's id, "-1" means there isn't a map point related with this key point
    std::vector<int> _relatedMpts;

    Sophus::SE3d _pose;

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