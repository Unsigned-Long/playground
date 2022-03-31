#ifndef BA_H
#define BA_H

#include "ceres/ceres.h"
#include "frame.h"

namespace ns_myslam {
  struct BundleAdjustment {
  private:
    Eigen::Vector2d _pixel;
    MonoCamera::Ptr _camera;

  public:
    BundleAdjustment(const Eigen::Vector2d &pixel, MonoCamera::Ptr camera)
        : _pixel(pixel), _camera(camera) {}

    // template <typename T>
    bool operator()(const double *const mapPt, const double *const pose, double *error) const {
      Eigen::Quaterniond q(pose);
      q.normalized();

      Eigen::Vector3d trans = Eigen::Vector3d(pose[4], pose[5], pose[6]);

      Sophus::SE3d tr(q, trans);

      Eigen::Vector3d ptMap(mapPt);
      Eigen::Vector3d ptCam = tr * ptMap;

      Eigen::Vector2d pixelRePrj = this->_camera->nplane2pixel(ptCam / ptCam(2));

      error[0] = this->_pixel(0) - pixelRePrj(0);
      error[1] = this->_pixel(1) - pixelRePrj(1);

      return true;
    }

    static ceres::CostFunction *createCostFun(const Eigen::Vector2d &pixel, MonoCamera::Ptr camera) {
      return new ceres::NumericDiffCostFunction<BundleAdjustment,ceres::CENTRAL, 2, 3, 7>(
          new BundleAdjustment(pixel, camera));
    }
  };
} // namespace ns_myslam

#endif