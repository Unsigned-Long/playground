#include "myslam.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

namespace ns_myslam {
  MySLAM::MySLAM(MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature)
      : _camera(camera), _orbFeature(orbFeature),
        _map(), _frames(),
        _initSystem(false), _initScale(false) {}

  MySLAM &MySLAM::addFrame(MatPtr grayImg) {
    ns_log::info("add a new frame");

    // construct a new frame
    timer.reBoot();

    // create the new frame
    Frame::Ptr curFrame = Frame::create(this->_frames.size(), grayImg, this->_camera, this->_orbFeature);

    if (this->_initSystem) {
      Frame::Ptr lastFrame = this->_frames.back();

      // find good matches
      auto goodMatches = this->findGoodMatches(lastFrame, curFrame);

      Eigen::Matrix3d rotMat_21;
      Eigen::Vector3d transMat_21;

      // ids of new map points need to be triangulated
      std::vector<int> triMptIds;
      // ids of new map points need to be estimate [PnP]
      std::vector<int> mptIdPnP;

      // create a map point for each good match if necessary
      for (const auto &match : goodMatches) {
        if (lastFrame->_relatedMpts.at(match.queryIdx) < 0) {
          // there is no map point for current match [key points pair]
          MapPoint::Ptr newMpt = MapPoint::create(this->_map.size(), Eigen::Vector3d());

          // record the key points to the map point
          newMpt->_frameFeatures.push_back({lastFrame->_id, match.queryIdx});
          newMpt->_frameFeatures.push_back({curFrame->_id, match.trainIdx});

          // add map point ptr to two frames
          lastFrame->_relatedMpts.at(match.queryIdx) = newMpt->_id;
          curFrame->_relatedMpts.at(match.trainIdx) = newMpt->_id;

          // add ptr to the main map
          this->_map.push_back(newMpt);

          triMptIds.push_back(newMpt->_id);
        } else {
          // there is a map point for key point in last frame
          int mptId = lastFrame->_relatedMpts.at(match.queryIdx);
          MapPoint::Ptr mpt = this->findMapPoint(mptId);

          mpt->_frameFeatures.push_back({curFrame->_id, match.trainIdx});
          curFrame->_relatedMpts.at(match.trainIdx) = mptId;

          mptIdPnP.push_back(mptId);
        }
      }

      // append the current frame
      this->_frames.push_back(curFrame);

      if (!this->_initScale) {
        ns_log::info("initialization for map scale [2d-2d]");

        timer.reStart();
        // init the scale [2d-2d] using the first two frames

        while (true) {
          // calculate the rotation matrix and translate matrix
          auto [R, t] = this->computeScale(goodMatches, lastFrame, curFrame);

          // opencv to eigen matrix
          cv::cv2eigen(R, rotMat_21);
          cv::cv2eigen(t, transMat_21);

          // find the badest match

          double maxError = this->matchError(goodMatches.front(), rotMat_21, transMat_21, lastFrame, curFrame);
          double meanError = maxError;
          int maxErrorIdx = 0;
          for (int i = 1; i != goodMatches.size(); ++i) {
            double error = this->matchError(goodMatches.at(i), rotMat_21, transMat_21, lastFrame, curFrame);
            meanError += error;
            if (error > maxError) {
              maxError = error;
              maxErrorIdx = i;
            }
          }
          meanError /= goodMatches.size();
          // if the error is big

          if (maxError < 10.0 * meanError) {
            break;
          }
          auto maxErrorMatch = goodMatches.at(maxErrorIdx);
          // remove the map point [invalid map point]
          int mptId = lastFrame->_relatedMpts.at(maxErrorMatch.queryIdx);
          this->findMapPoint(mptId) = nullptr;

          // remove the frame's  related map point
          lastFrame->_relatedMpts.at(maxErrorMatch.queryIdx) = -1;
          curFrame->_relatedMpts.at(maxErrorMatch.trainIdx) = -1;

          // remove the badest match
          goodMatches.at(maxErrorIdx) = goodMatches.back();
          goodMatches.pop_back();
        }

        // pose [from the first frame to current frame]
        curFrame->_pose_cw = Sophus::SE3d(rotMat_21, transMat_21) * lastFrame->_pose_cw;

        this->_initScale = true;

      } else {
        ns_log::info("run PnP estimate [3d-2d]");
        // pose from world to current frame
        auto pose = lastFrame->_pose_cw;

        while (true) {
          pose = this->estimatePnP(pose, mptIdPnP);

          double maxError = this->estimatePnPError(mptIdPnP.front(), pose);
          double meanError = maxError;
          int maxErrorIdx = 0;

          for (int i = 1; i != mptIdPnP.size(); ++i) {
            double error = this->estimatePnPError(mptIdPnP.at(i), pose);
            meanError += error;
            if (error > maxError) {
              maxError = error;
              maxErrorIdx = i;
            }
          }
          meanError /= mptIdPnP.size();

          if (maxError < 5.0 * meanError) {
            break;
          }

          // remove the match [key point with map point]
          int maxErrorId = mptIdPnP.at(maxErrorIdx);
          auto [frameId, kptIdx] = this->findMapPoint(maxErrorId)->_frameFeatures.back();
          this->findFrame(frameId)->_relatedMpts.at(kptIdx) = -1;
          // remove the related map point
          this->findMapPoint(maxErrorId)->_frameFeatures.pop_back();
          // remove the map point to be estimate [PnP]
          mptIdPnP.at(maxErrorIdx) = mptIdPnP.back();
          mptIdPnP.pop_back();
        }

        // pose from last frame to current frame
        auto pose_21 = pose * lastFrame->_pose_cw.inverse();
        // get rotation matrix and translate matrix
        rotMat_21 = pose_21.rotationMatrix();
        transMat_21 = pose_21.translation();

        curFrame->_pose_cw = pose;
      }

      ns_log::info("triangulation for map points");

      // triangulation
      while (true) {

        // find max error and mean error
        double triMaxError, triMeanError;
        int triMaxErrorIdx;

        for (int i = 0; i != triMptIds.size(); ++i) {
          // get map point
          auto mpt = this->findMapPoint(triMptIds.at(i));

          if (mpt == nullptr) {
            continue;
          }

          // get last two pixels
          auto [feature1, feature2] = mpt->lastTwoFramesFeatures();
          auto pixel1 = this->findKeyPoint(feature1.first, feature1.second);
          auto pixel2 = this->findKeyPoint(feature2.first, feature2.second);

          // triangulation
          auto [mptInLastFrame, error] = this->triangulation(
              Eigen::Vector2d(pixel1.pt.x, pixel1.pt.y),
              Eigen::Vector2d(pixel2.pt.x, pixel2.pt.y),
              rotMat_21, transMat_21);

          if (i == 0) {
            triMaxError = error;
            triMeanError = error;
            triMaxErrorIdx = 0;
          } else {
            triMeanError += error;
            if (triMaxError < error) {
              triMaxError = error;
              triMaxErrorIdx = i;
            }
          }

          // get coordinate value of map point in the world coordiante system
          mpt->_pt = lastFrame->_pose_cw.inverse() * mptInLastFrame;
        }

        triMeanError /= triMptIds.size();

        // if this frame is used to init the scale, then don't need to check the matches
        if (this->_frames.size() == 2) {
          break;
        }

        if (triMaxError < 10.0 * triMeanError) {
          break;
        }

        auto mpt = this->findMapPoint(triMptIds.at(triMaxErrorIdx));
        auto [feature1, feature2] = mpt->lastTwoFramesFeatures();
        this->findFrame(feature1.first)->_relatedMpts.at(feature1.second) = -1;
        this->findFrame(feature2.first)->_relatedMpts.at(feature2.second) = -1;
        mpt = nullptr;
        triMptIds.at(triMaxErrorIdx) = triMptIds.back();
        triMptIds.pop_back();
      }

    } else {
      ns_log::info("initialization for system");
      this->_initSystem = true;
      curFrame->_pose_cw = Sophus::SE3d();
      // append the current frame
      this->_frames.push_back(curFrame);
    }

    // output info
    ns_log::info("current frame pose [ref to world]");
    std::cout << this->currentPose().matrix3x4() << std::endl;
    std::cout << timer.total_elapsed("adding frame [" + std::to_string(curFrame->_id) + "] costs") << std::endl;
    std::cout << std::endl;

    this->showCurrentFrame(curFrame, 10);

    return *this;
  }

  std::vector<cv::DMatch> MySLAM::findGoodMatches(Frame::Ptr lastFrame, Frame::Ptr curFrame) {
    // compute matches
    auto matches = this->_orbFeature->match(lastFrame->_grayImg, lastFrame->_kpts, curFrame->_grayImg, curFrame->_kpts);
    // find good matches
    auto minDisIter = std::min_element(matches.cbegin(), matches.cend(), [](const cv::DMatch &m1, const cv::DMatch &m2) {
      return m1.distance < m2.distance;
    });
    std::vector<cv::DMatch> goodMatches;
    for (const auto &match : matches) {
      // the condition for good matches
      if (match.distance < std::max(2.0f * minDisIter->distance, 50.0f)) {
        goodMatches.push_back(match);
      }
    }
    return goodMatches;
  }

  std::pair<cv::Mat, cv::Mat> MySLAM::computeScale(const std::vector<cv::DMatch> &matches, Frame::Ptr frame1, Frame::Ptr frame2) {

    auto [pts1, pts2] = this->matchedKeyPoints(matches, frame1, frame2);

    // compute the essential matrix
    cv::Point2d principalPoint(_camera->cx, _camera->cy);
    double focalLength = (_camera->fx + _camera->fy) / 2.0f;
    cv::Mat essentialMatrix = findEssentialMat(pts1, pts2, focalLength, principalPoint);

    // get the rotation and translate matrix
    cv::Mat R, t;
    cv::recoverPose(essentialMatrix, pts1, pts2, R, t, focalLength, principalPoint);

    return {R, t};
  }

  Frame::Ptr &MySLAM::findFrame(int frameID) {
    return this->_frames.at(frameID);
  }

  MapPoint::Ptr &MySLAM::findMapPoint(int mptId) {
    return this->_map.at(mptId);
  }

  Eigen::Matrix3d MySLAM::vec2AntiMat(const Eigen::Vector3d &vec) {
    Eigen::Matrix3d mat;
    double x = vec(0), y = vec(1), z = vec(2);

    mat(0, 0) = 0.0, mat(0, 1) = -z, mat(0, 2) = y;
    mat(1, 0) = z, mat(1, 1) = 0.0, mat(1, 2) = -x;
    mat(2, 0) = -y, mat(2, 1) = x, mat(2, 2) = 0.0;

    return mat;
  }

  double MySLAM::matchError(const cv::DMatch &match,
                            const Eigen::Matrix3d &rotMat_21, const Eigen::Vector3d &transMat_21,
                            Frame::Ptr frame1, Frame::Ptr frame2) {
    auto p1 = frame1->_kpts.at(match.queryIdx).pt;
    auto x1 = this->_camera->pixel2nplane(Eigen::Vector2d(p1.x, p1.y));

    auto p2 = frame2->_kpts.at(match.trainIdx).pt;
    auto x2 = this->_camera->pixel2nplane(Eigen::Vector2d(p2.x, p2.y));

    double error = x2.transpose() * this->vec2AntiMat(transMat_21) * rotMat_21 * x1;

    return std::abs(error);
  }

  std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>>
  MySLAM::matchedKeyPoints(const std::vector<cv::DMatch> &matches, Frame::Ptr frame1, Frame::Ptr frame2) {
    // the point vector
    std::vector<cv::Point2d> pts1;
    std::vector<cv::Point2d> pts2;

    // find the matched pixel points
    for (const auto &match : matches) {
      pts1.push_back(frame1->_kpts.at(match.queryIdx).pt);
      pts2.push_back(frame2->_kpts.at(match.trainIdx).pt);
    }

    return {pts1, pts2};
  }

  std::pair<Eigen::Vector3d, double>
  MySLAM::triangulation(const Eigen::Vector2d &pixel1, const Eigen::Vector2d &pixel2,
                        const Eigen::Matrix3d &rotMat_21, const Eigen::Vector3d &transMat_21) {
    auto nplane1 = this->_camera->pixel2nplane(pixel1);
    auto nplane2 = this->_camera->pixel2nplane(pixel2);

    auto antiMat2 = this->vec2AntiMat(nplane2);

    Eigen::Vector3d l = -antiMat2 * transMat_21;
    Eigen::Vector3d B = antiMat2 * rotMat_21 * nplane1;

    double s1 = ((B.transpose() * B).inverse() * B.transpose() * l)(0, 0);

    double error = (nplane2.transpose() * this->vec2AntiMat(transMat_21) * rotMat_21 * nplane1).norm();

    return {nplane1 * s1, error};
  }

  cv::KeyPoint &MySLAM::findKeyPoint(int frameId, int kptIdx) {
    return this->findFrame(frameId)->_kpts.at(kptIdx);
  }

  Sophus::SE3d MySLAM::estimatePnP(const Sophus::SE3d &initVal, const std::vector<int> &mptIds) {
    Sophus::SE3d pose = initVal;

    // focal length in x and y direction of this camera
    double fx = this->_camera->fx;
    double fy = this->_camera->fy;

    for (int i = 0; i != 10; ++i) {
      Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Vector<double, 6> g = Eigen::Vector<double, 6>::Zero();

      for (const auto &mptId : mptIds) {

        auto mpt = this->findMapPoint(mptId);

        auto [frameId, kptIdx] = mpt->_frameFeatures.back();
        // key point
        auto kp = this->findKeyPoint(frameId, kptIdx);
        // map point
        auto mptPt = pose * mpt->_pt;
        double X = mptPt(0), Y = mptPt(1), Z = mptPt(2), ZInv = 1.0 / Z, ZInv2 = 1.0 / (Z * Z);

        Eigen::Vector2d pixel(kp.pt.x, kp.pt.y);

        // ocauculate the error
        auto error = pixel - this->_camera->nplane2pixel(Eigen::Vector3d(X / Z, Y / Z, 1.0));

        // organize the jacobian matrix
        Eigen::Matrix<double, 6, 2> J;
        J << fx * ZInv, 0.0,
            0.0, fy * ZInv,
            -fx * X * ZInv2, -fy * Y * ZInv2,
            -fx * X * Y * ZInv2, -fy - fy * Y * Y * ZInv2,
            fx + fx * X * X * ZInv2, fy * X * Y * ZInv2,
            -fx * Y * ZInv, fy * X * ZInv;
        J *= -1.0;

        H += J * J.transpose();
        g += -J * error;
      }
      // The first three components  represent the translational part , while the last three components
      // represents the rotation vector.
      Eigen::Vector<double, 6> deta = H.ldlt().solve(g);

      // update
      pose = Sophus::SE3d::exp(deta) * pose;

      if (deta.norm() < 1E-6) {
        break;
      }
    }
    return pose;
  }

  double MySLAM::estimatePnPError(int mptId, const Sophus::SE3d &pose_cw) {
    // find map point
    auto mptPt = this->findMapPoint(mptId)->_pt;
    auto mptCurCoord = pose_cw * mptPt;
    // reproject to pixel
    auto rePrjPixel = this->_camera->nplane2pixel(Eigen::Vector3d(
        mptCurCoord(0) / mptCurCoord(2), mptCurCoord(1) / mptCurCoord(2), 1.0));
    // find key point
    auto [frameId, kptIdx] = this->findMapPoint(mptId)->_frameFeatures.back();
    auto pixel = this->findKeyPoint(frameId, kptIdx);
    Eigen::Vector2d p(pixel.pt.x, pixel.pt.y);
    // compute error
    return (p - rePrjPixel).norm();
  }

  Sophus::SE3d MySLAM::currentPose() const {
    if (this->_frames.empty()) {
      return Sophus::SE3d();
    } else {
      return this->_frames.back()->_pose_cw.inverse();
    }
  }

  void MySLAM::showCurrentFrame(Frame::Ptr frame, int wait) const {
    cv::Mat img = frame->_grayImg->clone();
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    for (int i = 0; i != frame->_kpts.size(); ++i) {
      if (frame->_relatedMpts.at(i) >= 0) {
        auto kpt = frame->_kpts.at(i);
        cv::drawMarker(img, kpt.pt, cv::Scalar(0, 150, 0), cv::MarkerTypes::MARKER_TILTED_CROSS, 6, 2);
      }
    }
    cv::imshow("win", img);
    cv::waitKey(wait);
    return;
  }

} // namespace ns_myslam
