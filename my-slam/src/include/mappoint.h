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
    const int _id;
    // coordinate value of the map point
    Eigen::Vector3d _pt;
    // {witch frame, witch key point}
    std::vector<std::pair<int, int>> _frameFeatures;

  public:
    /**
     * @brief construct a new MapPoint
     *
     * @param mapPoint the coordinate value
     */
    MapPoint(int id, const Eigen::Vector3d &point);

    /**
     * @brief create a shared pointer
     */
    static MapPoint::Ptr create(int id, const Eigen::Vector3d &point);
  };

} // namespace ns_myslam

#endif