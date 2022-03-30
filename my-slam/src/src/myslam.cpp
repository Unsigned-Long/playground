#include "myslam.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"

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
      std::vector<int> mptIdTriangulate;

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

          mptIdTriangulate.push_back(newMpt->_id);
        } else {
          // there is a map point for key point in last frame
          int mptId = lastFrame->_relatedMpts.at(match.queryIdx);
          MapPoint::Ptr mpt = this->findMapPoint(mptId);
          mpt->_frameFeatures.push_back({curFrame->_id, match.trainIdx});
          curFrame->_relatedMpts.at(match.trainIdx) = mptId;
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
          auto [R, t] = this->initScale(goodMatches, lastFrame, curFrame);

          // opencv to eigen matrix
          cv::cv2eigen(R, rotMat_21);
          cv::cv2eigen(t, transMat_21);

          // find the badest match
          auto maxIter = std::max_element(
              goodMatches.begin(), goodMatches.end(),
              [this, lastFrame, curFrame, &rotMat_21, &transMat_21](const cv::DMatch &m1, const cv::DMatch &m2) {
                double error1 = this->matchError(m1, rotMat_21, transMat_21, lastFrame, curFrame);
                double error2 = this->matchError(m2, rotMat_21, transMat_21, lastFrame, curFrame);
                return error1 < error2;
              });

          // if the error is big
          double error = this->matchError(*maxIter, rotMat_21, transMat_21, lastFrame, curFrame);
          if (error < 0.05) {
            break;
          }

          // remove the map point [invalid map point]
          int mptId = lastFrame->_relatedMpts.at(maxIter->queryIdx);
          this->findMapPoint(mptId) = nullptr;

          // remove the frame's  related map point
          lastFrame->_relatedMpts.at(maxIter->queryIdx) = -1;
          curFrame->_relatedMpts.at(maxIter->trainIdx) = -1;

          // remove the badest match
          *maxIter = goodMatches.back();
          goodMatches.pop_back();
        }

        // pose [from the first frame to current frame]
        curFrame->_pose = Sophus::SE3d(rotMat_21, transMat_21) * lastFrame->_pose;

        this->_initScale = true;

      } else {
        ns_log::info("run PnP estimate [3d-2d]");
      }

      ns_log::info("triangulation for map points");
      // triangulation
      for (const auto &mapId : mptIdTriangulate) {
        // get map point
        auto mpt = this->findMapPoint(mapId);

        if (mpt == nullptr) {
          continue;
        }

        // get last two pixels
        auto [feature1, feature2] = mpt->lastTwoFramesFeatures();
        auto pixel1 = this->findKeyPoint(feature1.first, feature1.second);
        auto pixel2 = this->findKeyPoint(feature2.first, feature2.second);

        // triangulation
        auto mptInLastFrame = this->triangulation(
            Eigen::Vector2d(pixel1.pt.x, pixel1.pt.y),
            Eigen::Vector2d(pixel2.pt.x, pixel2.pt.y),
            rotMat_21, transMat_21);

        // get coordinate of map point in the space
        mpt->_pt = lastFrame->_pose.inverse() * mptInLastFrame;
      }

    } else {
      ns_log::info("initialization for system");
      this->_initSystem = true;
      curFrame->_pose = Sophus::SE3d();
      // append the current frame
      this->_frames.push_back(curFrame);
    }

    std::cout << timer.total_elapsed("adding frame [" + std::to_string(curFrame->_id) + "] costs") << std::endl;

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

  std::pair<cv::Mat, cv::Mat> MySLAM::initScale(const std::vector<cv::DMatch> &matches, Frame::Ptr frame1, Frame::Ptr frame2) {

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

  Eigen::Vector3d MySLAM::triangulation(const Eigen::Vector2d &pixel1, const Eigen::Vector2d &pixel2,
                                        const Eigen::Matrix3d &rotMat_21, const Eigen::Vector3d &transMat_21) {
    auto nplane1 = this->_camera->pixel2nplane(pixel1);
    auto nplane2 = this->_camera->pixel2nplane(pixel2);

    auto antiMat2 = this->vec2AntiMat(nplane2);

    Eigen::Vector3d l = -antiMat2 * transMat_21;
    Eigen::Vector3d B = antiMat2 * rotMat_21 * nplane1;

    double s1 = ((B.transpose() * B).inverse() * B.transpose() * l)(0, 0);

    return nplane1 * s1;
  }

  cv::KeyPoint &MySLAM::findKeyPoint(int frameId, int kptIdx) {
    return this->findFrame(frameId)->_kpts.at(kptIdx);
  }
} // namespace ns_myslam
