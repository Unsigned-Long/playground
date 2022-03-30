#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "eigen3/Eigen/Core"
#include "opencv2/features2d.hpp"

namespace ns_myslam {
  class MySLAM;

  class MapPoint {
  public:
    using Ptr = std::shared_ptr<MapPoint>;
    // {frame id, key point index}
    using kpt_pos_type = std::pair<int, int>;
    friend class MySLAM;

  private:
    const int _id;
    // coordinate value of the map point
    Eigen::Vector3d _pt;
    // {witch frame, witch key point}
    std::vector<kpt_pos_type> _frameFeatures;

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

    std::pair<kpt_pos_type, kpt_pos_type> lastTwoFramesFeatures();
  };

} // namespace ns_myslam

#endif