#include "frame.h"

namespace ns_myslam {
  Frame::Frame(int id, MatPtr grayImg, MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature)
      : _id(id), _grayImg(grayImg), _kpts(), _relatedMpts(), _pose_cw() {
    this->pretreatment(camera, orbFeature);
  }

  Frame::Ptr Frame::create(int id, MatPtr grayImg, MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature) {
    return std::make_shared<Frame>(id, grayImg, camera, orbFeature);
  }

  void Frame::pretreatment(MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature) {
    // undistort the image
    auto &[xRange, yRange] = camera->undistort(this->_grayImg);

    // detect the key points in the win range
    this->_kpts = orbFeature->detect(this->_grayImg, xRange, yRange);

    this->_relatedMpts.resize(this->_kpts.size());

    // select key point in win range
    for (int i = 0; i != this->_kpts.size(); ++i) {
      this->_kpts.at(i).pt.x += xRange.start;
      this->_kpts.at(i).pt.y += yRange.start;
      this->_relatedMpts.at(i) = -1;
    }
    return;
  }
} // namespace ns_myslam