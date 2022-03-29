#include "camera.h"

namespace ns_myslam {
  MonoCamera::MonoCamera(float fx, float fy, float cx, float cy,
                         float k1, float k2, float k3,
                         float p1, float p2)
      : fx(fx), fy(fy), fx_inv(1.0f / fx), fy_inv(1.0f / fy), cx(cx), cy(cy),
        k1(k1), k2(k2), k3(k3),
        p1(p1), p2(p2),
        _initialized(false), _winRange() {
    this->_cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    this->_distCoeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);
  }

  MonoCamera::Ptr MonoCamera::create(float fx, float fy, float cx, float cy, float k1, float k2, float k3, float p1, float p2) {
    return std::make_shared<MonoCamera>(fx, fy, cx, cy, k1, k2, k3, p1, p2);
  }

  Eigen::Vector3f MonoCamera::pixel2nplane(const Eigen::Vector2f &pixel) const {
    Eigen::Vector3f nplanePoint;
    nplanePoint[0] = (pixel[0] - this->cx) * this->fx_inv;
    nplanePoint[1] = (pixel[1] - this->cy) * this->fy_inv;
    nplanePoint[2] = 1.0f;
    return nplanePoint;
  }

  Eigen::Vector2f MonoCamera::nplane2pixel(const Eigen::Vector3f &nplanePoint) const {
    Eigen::Vector2f pixel;
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
        auto nplanePoint = this->pixel2nplane(Eigen::Vector2f(c, r));
        // undistort
        float x = nplanePoint[0], y = nplanePoint[1];
        float r = std::sqrt(x * x + y * y), r2 = r * r, r4 = r2 * r2, r6 = r4 * r2;
        Eigen::Vector3f dist;
        dist(0) = x * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
        dist(1) = y * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;
        dist(2) = 1.0f;
        // reproject
        Eigen::Vector2f pixel = this->nplane2pixel(dist);
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

      Eigen::Vector2f leftTop(0.0f, 0.0f), rightTop(cols - 1, 0.0f);
      Eigen::Vector2f leftBottom(0.0f, rows - 1), rightBottom(cols - 1, rows - 1);

      leftTop = this->findUndistortedPixel(leftTop), rightTop = this->findUndistortedPixel(rightTop);
      leftBottom = this->findUndistortedPixel(leftBottom), rightBottom = this->findUndistortedPixel(rightBottom);

      float xMin = std::max({leftTop(0), leftBottom(0), 0.0f});
      float xMax = std::min({rightTop(0), rightBottom(0), float(cols)});
      float yMin = std::max({leftTop(1), rightTop(1), 0.0f});
      float yMax = std::min({leftBottom(1), rightBottom(1), float(rows)});

      this->_winRange.first = cv::Range(xMin, xMax);
      this->_winRange.second = cv::Range(yMin, yMax);
      this->_initialized = true;
    }

    return this->_winRange;
  }

  bool MonoCamera::pixelInWinRange(const Eigen::Vector2f &pixel) const {
    if ((pixel[0] > this->_winRange.first.start && pixel[0] < this->_winRange.first.end) &&
        (pixel[1] > this->_winRange.second.start && pixel[1] < this->_winRange.second.end)) {
      return true;
    }
    return false;
  }

  Eigen::Vector2f MonoCamera::findUndistortedPixel(const Eigen::Vector2f &distortedPixel) const {
    // [50 times faster than OpenCV]
    Eigen::Vector2f undistortedPixel(distortedPixel);
    // do the gauss-newton methods
    for (int i = 0; i != 5; ++i) {
      auto nplanePoint = this->pixel2nplane(undistortedPixel);
      float x = nplanePoint[0], y = nplanePoint[1];
      float r = std::sqrt(x * x + y * y), r2 = r * r, r4 = r2 * r2, r6 = r4 * r2;
      // calculate the jacobian matrix
      float ju = -(1.0f + k1 * r2 + k2 * r4 + k3 * r6) - x * (2.0f * k1 * x + 4.0f * k2 * r2 * x + 6.0f * k3 * r4 * x) - 2.0f * p1 * y - 6.0f * p2 * x;
      float jv = -(1.0f + k1 * r2 + k2 * r4 + k3 * r6) - y * (2.0f * k1 * y + 4.0f * k2 * r2 * y + 6.0f * k3 * r4 * y) - 6.0f * p1 * y - 2.0f * p2 * x;
      // calculate reproject error
      Eigen::Vector3f dist;
      dist(0) = x * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
      dist(1) = y * (1.0f + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;
      dist(2) = 1.0f;
      Eigen::Vector2f error = distortedPixel - this->nplane2pixel(dist);
      // solve the equation
      Eigen::Vector2f deta(-error(0) / ju, -error(1) / jv);
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
