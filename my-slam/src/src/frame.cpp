#include "frame.h"

namespace ns_myslam {
  Frame::Frame(int id, MatPtr grayImg, MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature)
      : _id(id), _grayImg(grayImg), _keyPoints(), _mapPoints() {
    this->pretreatment(camera, orbFeature);
  }

  Frame::Ptr Frame::create(int id, MatPtr grayImg, MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature) {
    return std::make_shared<Frame>(id, grayImg, camera, orbFeature);
  }

  void Frame::pretreatment(MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature) {
    // undistort the image
    camera->undistort(this->_grayImg);
    // detect the key points
    auto keyPoints = orbFeature->detect(this->_grayImg);
    // select key point in win range
    for (auto kp : keyPoints) {
      if (camera->pixelInWinRange(Eigen::Vector2f(kp.pt.x, kp.pt.y))) {
        this->_keyPoints.push_back(kp);
      }
    }
    return;
  }
} // namespace ns_myslam