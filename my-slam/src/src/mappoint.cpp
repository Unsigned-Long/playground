#include "mappoint.h"

namespace ns_myslam {
  MapPoint::MapPoint(const Eigen::Vector3f &mapPoint) : _mapPoint(mapPoint), _frameFeatures() {}

  MapPoint::Ptr MapPoint::create(const Eigen::Vector3f &mapPoint) {
    return std::make_shared<MapPoint>(mapPoint);
  }
} // namespace ns_myslam
