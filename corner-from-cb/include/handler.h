#pragma once

#include <iostream>
#include "harris.h"
#include <string>
#include <list>
#include "pcl-1.12/pcl/kdtree/kdtree_flann.h"
#include "ceres/ceres.h"

namespace ns_test
{

    class CornerSelector
    {
    public:
        struct Linefitting
        {
            cv::Point2f _p;
            Linefitting(const cv::Point2f &p) : _p(p) {}

            bool operator()(const double *const params, double *out) const;
            // bool opertaor()(const double* const params, double *out) const;
        };

    private:
        CornerSelector() = default;
        /**
         * \brief Lets you select corners in the window with the mouse
         *        Select point: draw a rectangle with the left mouse button to select
         *        Delete selected point: right click
         */
        static void onMouse_select_corners(int event, int x, int y, int flags, void *userdata);

        /**
         * \brief Used to sort the selected corners by drawing a line with the mouse in the window
         *        Select a point: draw a line with the left mouse button
         */
        static void onMouse_sort_corners(int event, int x, int y, int flags, void *userdata);

        /**
         * \brief fitting the line by using ceres librarys
         */ 
        static cv::Point2f fittingLine(const std::vector<cv::Point2f> &points);

        // two points of the rect
        static cv::Point2f p1;
        static cv::Point2f p2;

        /**
         *  the points after decreasing number
         *  bool = true when is  selected
         */
        static std::list<std::pair<cv::Point2f, bool>> less;

        // kd-tree used to find near points fast
        static pcl::KdTreeFLANN<pcl::PointXY> kdtree;

        // the display image
        static cv::Mat dst;

        /**
         * \brief the fileds of the chess board
         */
        static int rows;
        static int cols;
        static float blockSize;

        // count the selected points' number
        static int selected;

        // the selected corners
        // the bool means isOrdered
        static std::vector<std::pair<cv::Point2f, bool>> corners;

        // the image point and the real world point
        static std::vector<std::pair<cv::Point2f, cv::Point2f>> mapping;

        // judge the mouse event when sort the points
        static bool isSorting;
        /**
         * \brief  this value control the precision of the corner detector
         *         greater value, more running time, high precision
         *         usually get 1.0 or 2.0
         */
    public:
        // main functions

        /**
         * \brief calculate the distance from the line[Ax + By + 1.0 = 0.0] to p
         */
        static double distance(const cv::Point2f &lineParams, const cv::Point2f &p);

        /**
         * \brief calculate the distance from the line[lp1, lp2] to p
         */
        static float distance(const cv::Point2f &lp1, const cv::Point2f &lp2, const cv::Point2f &p);
        /**
         * \brief search corner and create the mapping on the image
         * \param imagePath the chessboard image path
         * \param rows the rows of the chessboard
         * \param cols the cols of the chessboard
         * \param realBlockSize the size of the chessboard on real world
         * \param threshold the value to control the detected points' num in the harris algorithm
         * \param combineSize the size to combine in the kd-tree
         * \param isColor the display image type
         */
        static void doSearch(const std::string &imagePath, int rows, int cols, float realBlockSize, float threshold, float combineSize = 20, bool isColor = false);

        /**
         * \brief get the mapping
         *        [image pos], [real pos]
         */
        static std::vector<std::pair<cv::Point2f, cv::Point2f>> &getMapping();
    };
} // namespace ns_test
