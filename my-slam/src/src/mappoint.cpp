#include "mappoint.h"

namespace ns_myslam {
  MapPoint::MapPoint(int id, const Eigen::Vector3d &point)
      : _id(id), _pt(point), _frameFeatures() {}

  MapPoint::Ptr MapPoint::create(int id, const Eigen::Vector3d &point) {
    return std::make_shared<MapPoint>(id, point);
  }

  std::pair<MapPoint::kpt_pos_type, MapPoint::kpt_pos_type> MapPoint::lastTwoFramesFeatures() {
    auto feature1 = this->_frameFeatures.at(this->_frameFeatures.size() - 2);
    auto feature2 = this->_frameFeatures.at(this->_frameFeatures.size() - 1);
    return {feature1, feature2};
  }
} // namespace ns_myslam
