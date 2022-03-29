#include "camera.h"

namespace ns_myslam {
  MonoCamera::MonoCamera(double fx, double fy, double cx, double cy,
                         double k1, double k2, double k3,
                         double p1, double p2)
      : fx(fx), fy(fy), fx_inv(1.0f / fx), fy_inv(1.0f / fy), cx(cx), cy(cy),
        k1(k1), k2(k2), k3(k3),
        p1(p1), p2(p2),
        _initialized(false), _winRange() {
    this->_cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    this->_distCoeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);
  }

  MonoCamera::Ptr MonoCamera::create(double fx, double fy, double cx, double cy, double k1, double k2, double k3, double p1, double p2) {
    return std::make_shared<MonoCamera>(fx, fy, cx, cy, k1, k2, k3, p1, p2);
  }

  Eigen::Vector3d MonoCamera::pixel2nplane(const Eigen::Vector2d &pixel) const {
    Eigen::Vector3d nplanePoint;
    nplanePoint[0] = (pixel[0] - this->cx) * this->fx_inv;
    nplanePoint[1] = (pixel[1] - this->cy) * this->fy_inv;
    nplanePoint[2] = 1.0f;
    return nplanePoint;
  }

  Eigen::Vector2d MonoCamera::nplane2pixel(const Eigen::Vector3d &nplanePoint) const {
    Eigen::Vector2d pixel;
    pixel[0] = nplanePoint[0] * this->fx + this->cx;
    pixel[1] = nplanePoint[1] * this->fy + this->cy;
    return pixel;
  }

  const std::pair<cv::Range, cv::Range> &MonoCamera::undistort(MatPtr &grayImg) const {
    // rows and cols of this image
    int rows = grayImg->rows, cols = grayImg->cols;
    // image to keep the undistorted image
    auto undistortedImg = std::make_shared<cv::Mat>(rows, cols, grayImg->type());
    // undistort [2 times faster than OpenCV]
    for (int r = 0; r != rows; ++r) {
      auto dstPtr = undistortedImg->ptr<uchar>(r);
      for (int c = 0; c != cols; ++c) {
        // project
        auto nplanePoint = this->pixel2nplane(Eigen::Vector2d(c, r));
        // undistort
        double x = nplanePoint[0], y = nplanePoint[1];
        double r = std::sqrt(x * x + y * y), r2 = r * r, r4 = r2 * r2, r6 = r4 * r2;
        Eigen::Vector3d dist;
        dist(0) = x * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
        dist(1) = y * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;
        dist(2) = 1.0f;
        // reproject
        Eigen::Vector2d pixel = this->nplane2pixel(dist);
        int rDist = pixel(1), cDist = pixel(0);
        // assign
        if (rDist < 0 || rDist > rows - 1 || cDist < 0 || cDist > cols - 1) {
          dstPtr[c] = 0;
        } else {
          dstPtr[c] = grayImg->ptr<uchar>(rDist)[cDist];
        }
      }
    }
    grayImg.swap(undistortedImg);

    // calculate the range of the image
    if (!this->_initialized) {

      Eigen::Vector2d leftTop(0.0f, 0.0f), rightTop(cols - 1, 0.0f);
      Eigen::Vector2d leftBottom(0.0f, rows - 1), rightBottom(cols - 1, rows - 1);

      leftTop = this->findUndistortedPixel(leftTop), rightTop = this->findUndistortedPixel(rightTop);
      leftBottom = this->findUndistortedPixel(leftBottom), rightBottom = this->findUndistortedPixel(rightBottom);

      double xMin = std::max({leftTop(0), leftBottom(0), 0.0});
      double xMax = std::min({rightTop(0), rightBottom(0), double(cols)});
      double yMin = std::max({leftTop(1), rightTop(1), 0.0});
      double yMax = std::min({leftBottom(1), rightBottom(1), double(rows)});

      this->_winRange.first = cv::Range(xMin, xMax);
      this->_winRange.second = cv::Range(yMin, yMax);
      this->_initialized = true;
    }

    return this->_winRange;
  }

  bool MonoCamera::pixelInWinRange(const Eigen::Vector2d &pixel) const {
    if ((pixel[0] > this->_winRange.first.start && pixel[0] < this->_winRange.first.end) &&
        (pixel[1] > this->_winRange.second.start && pixel[1] < this->_winRange.second.end)) {
      return true;
    }
    return false;
  }

  Eigen::Vector2d MonoCamera::findUndistortedPixel(const Eigen::Vector2d &distortedPixel) const {
    // [50 times faster than OpenCV]
    Eigen::Vector2d undistortedPixel(distortedPixel);
    // do the gauss-newton methods
    for (int i = 0; i != 5; ++i) {
      auto nplanePoint = this->pixel2nplane(undistortedPixel);
      double x = nplanePoint[0], y = nplanePoint[1];
      double r = std::sqrt(x * x + y * y), r2 = r * r, r4 = r2 * r2, r6 = r4 * r2;
      // calculate the jacobian matrix
      double ju = -(1.0f + k1 * r2 + k2 * r4 + k3 * r6) - x * (2.0f * k1 * x + 4.0f * k2 * r2 * x + 6.0f * k3 * r4 * x) - 2.0f * p1 * y - 6.0f * p2 * x;
      double jv = -(1.0f + k1 * r2 + k2 * r4 + k3 * r6) - y * (2.0f * k1 * y + 4.0f * k2 * r2 * y + 6.0f * k3 * r4 * y) - 6.0f * p1 * y - 2.0f * p2 * x;
      // calculate reproject error
      Eigen::Vector3d dist;
      dist(0) = x * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
      dist(1) = y * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;
      dist(2) = 1.0f;
      Eigen::Vector2d error = distortedPixel - this->nplane2pixel(dist);
      // solve the equation
      Eigen::Vector2d deta(-error(0) / ju, -error(1) / jv);
      undistortedPixel += deta;
      if (deta.norm() < 1.5f) {
        break;
      }
    }
    return undistortedPixel;
  }

  std::ostream &operator<<(std::ostream &os, const MonoCamera &c) {
    os << "{";
    os << "'focal': [" << c.fx << ", " << c.fy << "], ";
    os << "'center': [" << c.cx << ", " << c.cy << "], ";
    os << "'radDist': [" << c.k1 << ", " << c.k2 << ", " << c.k3 << "], ";
    os << "'tanDist': [" << c.p1 << ", " << c.p2 << "]}";
    return os;
  }
} // namespace ns_myslam
