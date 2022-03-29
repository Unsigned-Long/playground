#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "eigen3/Eigen/Core"
#include "opencv2/features2d.hpp"

namespace ns_myslam {
  class MySLAM;

  class MapPoint {
  public:
    using Ptr = std::shared_ptr<MapPoint>;
    friend class MySLAM;

  private:
    Eigen::Vector3f _mapPoint;
    std::unordered_map<int, int> _frameFeatures;

  public:
    MapPoint(const Eigen::Vector3f &mapPoint);

    static MapPoint::Ptr create(const Eigen::Vector3f &mapPoint);
  };

} // namespace ns_myslam

#endif