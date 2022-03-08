#pragma once

#include <iostream>
#include <array>
#include <list>
#include <algorithm>
#include <ceres/ceres.h>
#include <pcl-1.12/pcl/kdtree/kdtree_flann.h>
#include <eigen3/Eigen/Dense>
#include <pcl-1.12/pcl/kdtree/kdtree_flann.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

namespace ns_test
{

#pragma region params
    namespace params
    {
        /**
         * \brief the init size of the prototypes
         *        large image uses large size
         *        possible choices : [11, 17, 21]
         */
        constexpr int initProtoSize = 25;
        /**
         * \brief the value of 150 is the low boundary for the chessboard corner candidates,
         *        larger value means find less candidates.
         *        Change it if necessary.
         */
        constexpr int likeHoodThreshold = 120;
        /**
         * \brief the value of [0.66] is a threshold for erase candidates,
         *        smaller value means erasing more.
         *        value range : (0.0, 1.0)
         *        Change it if necessary.
         */
        constexpr double edgeOrientationsThreshold = 0.8;
        /**
         * \brief the value of [0.0] is the threshold for the selected coeners
         *        larger value means selected less points.
         *        Change it if necessary.
         */
        constexpr double registerCornersThreshold = 0.0;
        /**
         * \brief the value control the process grow range of the chessboard
         *        large value means more dangers, decide it based on the image.
         */ 
        constexpr int grawChessBoardThreshold = 1000;
    }

#pragma region static class Gaussian
    template <typename _Ty>
    class Gaussian
    {
    public:
        using value_type = _Ty;

    private:
        Gaussian() = default;

    public:
        // the max value is sigma * std::sqrt(2.0 * M_PI)
        static value_type gaussian(value_type x, value_type mean, value_type sigma);
        // the max value is 1
        static value_type gaussianNormalized(value_type x, value_type mean, value_type sigma);
    };

    template <typename _Ty>
    typename Gaussian<_Ty>::value_type Gaussian<_Ty>::gaussian(value_type x, value_type mean, value_type sigma)
    {
        return Gaussian<_Ty>::gaussianNormalized(x, mean, sigma) / (sigma * std::sqrt(2.0 * M_PI));
    }

    template <typename _Ty>
    typename Gaussian<_Ty>::value_type Gaussian<_Ty>::gaussianNormalized(value_type x, value_type mean, value_type sigma)
    {
        return std::pow(M_E, -0.5 * (std::pow((x - mean) / sigma, 2)));
    }
#pragma endregion

#pragma region class CBDector
    /**
     * \brief the main class to find chessboard corners
     */
    class CBDector
    {
    public:
        using cornerPrototype = std::array<cv::Mat, 4>;

    private:
        // CV_8UC3
        cv::Mat _imgSrc;
        // CV_8UC1
        cv::Mat _imgGray;
        // CV_8UC1
        cv::Mat _imgLikehood;
        // CV_32FC1
        cv::Mat _imgAngle;
        // CV_32FC1
        cv::Mat _imgWeight;
        /**
         *  \brief two template types to convolute with image
         *  _type1 : horizontal direction
         *  _type2 : Tilt 45 degrees direction
         */
        cornerPrototype _type1;
        cornerPrototype _type2;

        /**
         * \brief the structure carrying corner information
         */
        struct ChessCorner
        {
            // the pixel position
            cv::Point _pos;
            // the gray grad of this position
            cv::Point2f _grad;
        };

        /**
         * \brief a structure to graw the hold chessboard
         */
        struct ChessBoard
        {
            // the size of the chessboard
            int _rows;
            int _cols;

            std::vector<cv::Point2f> _corners;

            ChessBoard() = delete;

            ChessBoard(int r, int c) : _rows(r), _cols(c), _corners(r * c, cv::Point2f()) {}

            // get one element at the position
            const cv::Point2f &operator()(int r, int c) const;

            cv::Point2f &operator()(int r, int c);

            // add one row on top
            void addRowUp();

            // add one row on bottom
            void addRowDown();

            // add one col at left
            void addColLeft();

            // add one col at right
            void addColRight();
        };

        // the init chessboard
        ChessBoard _chessboard = ChessBoard(3, 3);

        // position[x, y] and the two main direction
        std::list<ChessCorner> _candidates;

        // the coeners
        std::vector<ChessCorner> _corners;

        // the center of the chessboard
        ChessCorner _center;

        /**
         *  a structure to organize the result
         */
        struct Mapping
        {
            cv::Point2f _pixel;
            cv::Point2f _real;
        };

        // the result
        std::vector<Mapping> _mapping;

    public:
        CBDector() = delete;

        // read the image and init this algorithm
        CBDector(const std::string &imagePath) : _imgSrc(cv::imread(imagePath, cv::IMREAD_COLOR)) { this->init(); }

        // run the algorithm
        void process(float blockSize, std::string imgOutputDir = "");

        // get the result
        std::vector<Mapping> &getMapping() { return this->_mapping; }

    private:
        // calculate the direction from start to end
        static float direction(const cv::Point2f &start, const cv::Point2f &end);

        // judgement for angle range
        static bool isInRange(float start_radian, float end_radian, float target);

        // calculate the distance between p1 and p2
        static float distance(const cv::Point2f &p1, const cv::Point2f &p2);

        // it's used in the init function to create two kinds of proto template
        static cv::Mat createCornerPrototype(float start_radian, float end_radian, int size);

        // output something on the console
        static void outputInConsole(const std::string &words);

        // use mean shift algorithm to find two extremums
        static std::pair<int, int> findModesMeanShift(const std::array<float, 32> &hist);

        // translate the grad[radian] to a direction vector
        static cv::Vec2f gradVector(float grad);

        // calculate the energy of this chessboard
        float chessboard_energy(const ChessBoard &chessboard);

        // find corner at the director
        ChessCorner &findCorner(const cv::Vec2f &dir, const ChessCorner &p);

        // direction is [left]
        ChessCorner &findCornerLeft(const ChessCorner &center);

        // direction is [right]
        ChessCorner &findCornerRight(const ChessCorner &center);

        // direction is [up]
        ChessCorner &findCornerUp(const ChessCorner &center);

        // direction is [down]
        ChessCorner &findCornerDown(const ChessCorner &center);

        // to predict the corners
        pcl::PointXY predictCorners(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Point2f &p3);

        // calculate the likehood based on the gray image
        void calCornerLikehood();

        // use NMS algorithm to erase some bad corner candidates
        void nonMaximumSuppression();

        // calculate  the edge orientation
        void edgeOrientations();

        // register corners
        void registerCorners();

        // graw the chessboard
        void grawChessBoard();

        // get the mapping relationship
        void organizationMapping(float blockSize);

        // init this algorithm
        void init();
    };
#pragma endregion
} // namespace ns_test
