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
    // construct a new frame
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
        // init the scale [2d-2d] using the first two frames
        auto [R, t] = this->initScale();
        Eigen::Matrix3d rotMatrix;
        Eigen::Vector3d transMatrix;
        cv::cv2eigen(R, rotMatrix), cv::cv2eigen(t, transMatrix);
        curFrame->_pose = Sophus::SE3d(rotMatrix, transMatrix);
        this->_initScale = true;
      } else {
        // run [3d-2d]
      }
    } else {
      this->_initSystem = true;
      curFrame->_pose = Sophus::SE3d();
      // append the current frame
      this->_frames.push_back(curFrame);
      // {frameId, the index in vector "_frames"}
      this->_framesIdxMap.insert({curFrame->_id, this->_frames.size() - 1});
    }
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
      if (match.distance < std::max(2.0f * minDisIter->distance, 25.0f)) {
        goodMatches.push_back(match);
      }
    }
    return goodMatches;
  }

  std::pair<cv::Mat, cv::Mat> MySLAM::initScale() {
    std::vector<cv::Point2f> pts1(this->_map.size());
    std::vector<cv::Point2f> pts2(this->_map.size());
    int count = 0;
    for (const auto &mpt : this->_map) {
      // the first frame
      auto [frameId1, kptIdx1] = mpt->_frameFeatures.at(0);
      // the second frame
      auto [frameId2, kptIdx2] = mpt->_frameFeatures.at(1);
      // insert points
      pts1.at(count) = this->findFrame(frameId1)->_kpts.at(kptIdx1).pt;
      pts2.at(count) = this->findFrame(frameId2)->_kpts.at(kptIdx2).pt;
      ++count;
    }
    // compute the essential matrix
    cv::Point2d principalPoint(_camera->cx, _camera->cy);
    double focalLength = (_camera->fx + _camera->fy) / 2.0f;
    cv::Mat essentialMatrix = findEssentialMat(pts1, pts2, focalLength, principalPoint);
    // get the rotation and translate matrix
    cv::Mat R, t;
    cv::recoverPose(essentialMatrix, pts1, pts2, R, t, focalLength, principalPoint);
    return {R, t};
  }

  Frame::Ptr MySLAM::findFrame(int frameID) {
    return this->_frames.at(this->_framesIdxMap.at(frameID));
  }

  MapPoint::Ptr MySLAM::findMapPoint(int keyPointId) {
    return this->_map.at(this->_mptIdxMap.at(keyPointId));
  }
} // namespace ns_myslam
