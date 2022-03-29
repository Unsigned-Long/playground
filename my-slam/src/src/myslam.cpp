#include "myslam.h"
#include "artwork/timer/timer.h"

namespace ns_myslam {
  MySLAM::MySLAM(MonoCamera::Ptr camera, ORBFeature::Ptr orbFeature)
      : _camera(camera), _orbFeature(orbFeature), _map(), _frames(),
        _initSystem(false), _initScale(false) {}

  MySLAM &MySLAM::addFrame(int id, MatPtr grayImg) {
    // construct a new frame
    Frame::Ptr curFrame = Frame::create(id, grayImg, this->_camera, this->_orbFeature);
    // if this system is initialized, than do the front end
    if (this->_initSystem) {
      Frame::Ptr lastFrame = this->_frames.back();
      // find good matches
      auto goodMatches = this->findGoodMatches(lastFrame, curFrame);
      // create a map point for each good match if necessary
      for (const auto &match : goodMatches) {
        auto iter = lastFrame->_mapPoints.find(match.queryIdx);
        if (iter == lastFrame->_mapPoints.cend()) {
          // there is no map point for current match [key points pair]
          MapPoint::Ptr newMapPoint = MapPoint::create(Eigen::Vector3f());

          // record the key points to the map point
          newMapPoint->_frameFeatures.insert({lastFrame->_id, match.queryIdx});
          newMapPoint->_frameFeatures.insert({curFrame->_id, match.trainIdx});

          // add map point ptr to two frames
          lastFrame->_mapPoints.insert({match.queryIdx, newMapPoint});
          curFrame->_mapPoints.insert({match.trainIdx, newMapPoint});
          // add ptr to the main map
          this->_map.push_back(newMapPoint);
        } else {
          // there is a map point for key point in last frame
          MapPoint::Ptr mapPoint = iter->second;
          mapPoint->_frameFeatures.insert({curFrame->_id, match.trainIdx});
          curFrame->_mapPoints.insert({match.trainIdx, mapPoint});
        }
      }
      if (!this->_initScale) {
        // init the scale [2d-2d]
        this->_initScale = true;
      } else {
        // run [3d-2d]
      }
    }
    if (!this->_initSystem) {
      this->_initSystem = true;
    }
    // append the current frame
    this->_frames.push_back(curFrame);
    return *this;
  }

  std::vector<cv::DMatch> MySLAM::findGoodMatches(Frame::Ptr lastFrame, Frame::Ptr curFrame) {
    // compute matches
    auto matches = this->_orbFeature->match(lastFrame->_grayImg, lastFrame->_keyPoints, curFrame->_grayImg, curFrame->_keyPoints);
    // find good matches
    auto minDisIter = std::min_element(matches.cbegin(), matches.cend(), [](const cv::DMatch &m1, const cv::DMatch &m2) {
      return m1.distance < m2.distance;
    });
    std::vector<cv::DMatch> goodMatches;
    for (const auto &match : matches) {
      // the condition fot good matches
      if (match.distance < std::max(2.0f * minDisIter->distance, 30.0f)) {
        goodMatches.push_back(match);
      }
    }
    return goodMatches;
  }
} // namespace ns_myslam
