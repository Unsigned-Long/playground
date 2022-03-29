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

    bool _initSystem;
    bool _initScale;

  public:
    MySLAM(MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature);

    MySLAM &addFrame(int id, MatPtr grayImg);

  protected:
    std::vector<cv::DMatch> findGoodMatches(Frame::Ptr lastFrame, Frame::Ptr curFrame);
  };
} // namespace ns_myslam

#endif