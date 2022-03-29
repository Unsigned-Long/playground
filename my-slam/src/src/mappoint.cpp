#include "mappoint.h"

namespace ns_myslam {
  MapPoint::MapPoint(int id, const Eigen::Vector3d &point)
      : _id(id), _pt(point), _frameFeatures() {}

  MapPoint::Ptr MapPoint::create(int id, const Eigen::Vector3d &point) {
    return std::make_shared<MapPoint>(id, point);
  }
} // namespace ns_myslam
