#include "myslam.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"

namespace ns_myslam {
  MySLAM::MySLAM(MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature)
      : _camera(camera), _orbFeature(orbFeature),
        _map(), _frames(),
        _mptIdxMap(), _framesIdxMap(),
        _initSystem(false), _initScale(false) {}

  MySLAM &MySLAM::addFrame(int id, MatPtr grayImg) {
    ns_log::info("add a new frame");

    // construct a new frame
    timer.reBoot();
    Frame::Ptr curFrame = Frame::create(id, grayImg, this->_camera, this->_orbFeature);

    // if this system is initialized, than match features
    if (this->_initSystem) {
      Frame::Ptr lastFrame = this->_frames.back();

      // find good matches
      auto goodMatches = this->findGoodMatches(lastFrame, curFrame);

      // create a map point for each good match if necessary
      for (const auto &match : goodMatches) {
        if (lastFrame->_relatedMpts.at(match.queryIdx) < 0) {
          // there is no map point for current match [key points pair]
          MapPoint::Ptr newMpt = MapPoint::create(this->_map.size() - 1, Eigen::Vector3d());

          // record the key points to the map point
          newMpt->_frameFeatures.push_back({lastFrame->_id, match.queryIdx});
          newMpt->_frameFeatures.push_back({curFrame->_id, match.trainIdx});

          // add map point ptr to two frames
          lastFrame->_relatedMpts.at(match.queryIdx) = newMpt->_id;
          curFrame->_relatedMpts.at(match.trainIdx) = newMpt->_id;

          // add ptr to the main map
          this->_map.push_back(newMpt);
          this->_mptIdxMap.insert({newMpt->_id, this->_map.size() - 1});
        } else {
          // there is a map point for key point in last frame
          int mptId = lastFrame->_relatedMpts.at(match.queryIdx);
          MapPoint::Ptr mapPoint = this->findMapPoint(mptId);
          mapPoint->_frameFeatures.push_back({curFrame->_id, match.trainIdx});
          curFrame->_relatedMpts.at(match.trainIdx) = mptId;
        }
      }

      // append the current frame
      this->_frames.push_back(curFrame);
      // {frameId, the index in vector "_frames"}
      this->_framesIdxMap.insert({curFrame->_id, this->_frames.size() - 1});

      if (!this->_initScale) {
        ns_log::info("initialization for map scale");

        timer.reStart();
        // init the scale [2d-2d] using the first two frames

        Eigen::Matrix3d rotMat;
        Eigen::Vector3d transMat;

        while (true) {
          // calculate the rotation matrix and translate matrix
          auto [R, t] = this->initScale(goodMatches, lastFrame, curFrame);

          // opencv to eigen matrix
          cv::cv2eigen(R, rotMat);
          cv::cv2eigen(t, transMat);

          // find the badest match
          auto maxIter = std::max_element(goodMatches.begin(), goodMatches.end(),
                                          [this, lastFrame, curFrame, &rotMat, &transMat](const cv::DMatch &m1, const cv::DMatch &m2) {
                                            double error1 = this->matchError(m1, rotMat, transMat, lastFrame, curFrame);
                                            double error2 = this->matchError(m2, rotMat, transMat, lastFrame, curFrame);
                                            return error1 < error2;
                                          });

          // if the error is big
          double error = this->matchError(*maxIter, rotMat, transMat, lastFrame, curFrame);
          if (error < 0.05) {
            break;
          }

          // remove the map point
          int mptId = lastFrame->_relatedMpts.at(maxIter->queryIdx);
          this->findMapPoint(mptId) = nullptr;

          // remove the frame's  related map point
          lastFrame->_relatedMpts.at(maxIter->queryIdx) = -1;
          curFrame->_relatedMpts.at(maxIter->trainIdx) = -1;

          // remove the badest match
          *maxIter = goodMatches.back();
          goodMatches.pop_back();
        }

        curFrame->_pose = Sophus::SE3d(rotMat, transMat) * lastFrame->_pose;

        this->_initScale = true;

      } else {
        ns_log::info("run [3d-2d] estimate");
        
      }
    } else {
      ns_log::info("initialization for system");
      this->_initSystem = true;
      curFrame->_pose = Sophus::SE3d();
      // append the current frame
      this->_frames.push_back(curFrame);
      // {frameId, the index in vector "_frames"}
      this->_framesIdxMap.insert({curFrame->_id, this->_frames.size() - 1});
    }

    std::cout << timer.total_elapsed("adding frame [" + std::to_string(id) + "] costs") << std::endl;

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

  std::pair<cv::Mat, cv::Mat> MySLAM::initScale(const std::vector<cv::DMatch> &matches, Frame::Ptr firFrame, Frame::Ptr sedFrame) {

    auto [pts1, pts2] = this->matchedKeyPoints(matches, firFrame, sedFrame);

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
    return this->_frames.at(this->_framesIdxMap.at(frameID));
  }

  MapPoint::Ptr &MySLAM::findMapPoint(int keyPointId) {
    return this->_map.at(this->_mptIdxMap.at(keyPointId));
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
                            const Eigen::Matrix3d &rotMat, const Eigen::Vector3d &transMat,
                            Frame::Ptr firFrame, Frame::Ptr sedFrame) {
    auto p1 = firFrame->_kpts.at(match.queryIdx).pt;
    auto x1 = this->_camera->pixel2nplane(Eigen::Vector2d(p1.x, p1.y));

    auto p2 = sedFrame->_kpts.at(match.trainIdx).pt;
    auto x2 = this->_camera->pixel2nplane(Eigen::Vector2d(p2.x, p2.y));

    double error = x2.transpose() * this->vec2AntiMat(transMat) * rotMat * x1;

    return std::abs(error);
  }

  std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>>
  MySLAM::matchedKeyPoints(const std::vector<cv::DMatch> &matches, Frame::Ptr firFrame, Frame::Ptr sedFrame) {
    // the point vector
    std::vector<cv::Point2d> pts1;
    std::vector<cv::Point2d> pts2;

    // find the matched pixel points
    for (const auto &match : matches) {
      pts1.push_back(firFrame->_kpts.at(match.queryIdx).pt);
      pts2.push_back(sedFrame->_kpts.at(match.trainIdx).pt);
    }

    return {pts1, pts2};
  }

} // namespace ns_myslam
