#ifndef CAMERA_H
#define CAMERA_H

#include "eigen3/Eigen/Core"
#include "opencv2/core.hpp"
#include <iostream>
#include <memory>

namespace ns_myslam {
  using MatPtr = std::shared_ptr<cv::Mat>;

  class MonoCamera {
  public:
    using Ptr = std::shared_ptr<MonoCamera>;

  public:
    const double fx, fy, fx_inv, fy_inv;
    const double cx, cy;
    const double k1, k2, k3;
    const double p1, p2;

  private:
    cv::Mat _cameraMatrix;
    cv::Mat _distCoeffs;

    mutable bool _initialized;

    const bool _hasDistortion;

    // {[xMin, xMax), [yMin, yMax)}
    mutable std::pair<cv::Range, cv::Range> _winRange;

  public:
    MonoCamera() = delete;

    /**
     * @brief construct a new mono-camera
     *
     * @param fx focal length in x-direction
     * @param fy focal length in y-direction
     * @param cx x coordinate of camera optical center in pixel coordinate system
     * @param cy y coordinate of camera optical center in pixel coordinate system
     * @param k1 radial distortion parameter
     * @param k2 radial distortion parameter
     * @param k3 radial distortion parameter
     * @param p1 tangential distortion parameter
     * @param p2 tangential distortion parameter
     */
    MonoCamera(double fx, double fy, double cx, double cy, double k1, double k2, double k3, double p1, double p2);

    MonoCamera(double fx, double fy, double cx, double cy);

    static MonoCamera::Ptr create(double fx, double fy, double cx, double cy, double k1, double k2, double k3, double p1, double p2);

    static MonoCamera::Ptr create(double fx, double fy, double cx, double cy);

  public:
    /**
     * @brief project the points on the pixel coordinate plane onto the normalized pixel coordinate plane
     *
     * @param pixel points on the pixel coordinate plane
     * @return Eigen::Vector3d coordinates of points on the normalized pixel coordinate plane
     */
    Eigen::Vector3d pixel2nplane(const Eigen::Vector2d &pixel) const;

    /**
     * @brief project the points on the normalized pixel coordinate plane onto the pixel coordinate plane
     *
     * @param nplanePoint points on the normalized pixel coordinate plane
     * @return Eigen::Vector2d coordinates of points on the pixel coordinate plane
     */
    Eigen::Vector2d nplane2pixel(const Eigen::Vector3d &nplanePoint) const;

    /**
     * @brief undistort an image
     *
     * @param grayImg the gray image to be undistorted
     * @return std::pair<cv::Range, cv::Range> the (minu, maxu) and (minv, maxv) of the valid boundary
     */
    const std::pair<cv::Range, cv::Range> &undistort(MatPtr &grayImg) const;

    /**
     * @brief whether the pixel is in the window range
     *
     * @param pixel the pixel
     * @return true it's in the win range
     * @return false it's not in the win range
     */
    bool pixelInWinRange(const Eigen::Vector2d &pixel) const;

  protected:
    /**
     * @brief compute the undistorted pixel
     *
     * @param distortedPixel the distorted pixel
     * @return Eigen::Vector2d the undistorted pixel
     */
    Eigen::Vector2d findUndistortedPixel(const Eigen::Vector2d &distortedPixel) const;
  };

  /**
   * @brief outptu the info of the mono-camera
   */
  std::ostream &operator<<(std::ostream &os, const MonoCamera &c);
} // namespace ns_myslam

#endif