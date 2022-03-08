#pragma once

#include <iostream>
#include <string>
#include <ceres/ceres.h>
#include "point.h"
#include <eigen3/Eigen/Dense>

namespace ns_test
{
    class Camera
    {
    private:
#pragma region inner structures
        /**
         * \brief a structure to organize data
         */
        struct Mapping
        {
            Point2d _pixel;
            Point2d _real;
            Mapping() = default;
            Mapping(const Point2d &pixel, const Point2d &real) : _pixel(pixel), _real(real) {}
        };
        /**
         * \brief a structure to organize data
         */
        struct ImgData
        {
            std::vector<Mapping> _mappingVec;

            Eigen::Matrix3d _H;

            Eigen::Matrix3d _Rt;

            ImgData() = default;
        };

        /**
         * \brief a ceres structure to calculate the params matrix
         */
        struct Ceres_Params
        {
            const Mapping *_map;

            Ceres_Params(const Mapping *map) : _map(map) {}
            /**
             * \brief params[8] out[2]
             */
            bool operator()(const double *const params, double *out) const;
        };

        /**
         * \brief a ceres structure to calculate the inner params matrix
         */
        struct Ceres_InnerParams
        {
            const Eigen::Matrix3d *_H;

            Ceres_InnerParams(const Eigen::Matrix3d *H) : _H(H) {}

            /**
             * \brief params[6] out[2]
             */
            bool operator()(const double *const params, double *out) const;

            Eigen::MatrixXd helper(int i, int j) const;
        };

        /**
         * \brief a ceres structure to do Levenberg-Marquardt
         */
        struct Ceres_ML
        {
            const Mapping *_map;

            Ceres_ML(const Mapping *map) : _map(map) {}
            /**
             * \brief params[5 + 6] out[2]
             */
            bool operator()(const double *const innerParams, const double *const outerParams, double *out) const;
        };

        /**
         * \brief a ceres structure to calculate the distortion Params
         */
        struct Ceres_DistortionParams
        {
            const Eigen::MatrixXd _xy;
            const Point2d *_disPixel;
            const Eigen::Matrix3d *_A;

            Ceres_DistortionParams(const Eigen::MatrixXd &xy, const Point2d &disPixel, const Eigen::Matrix3d &A)
                : _xy(xy), _disPixel(&disPixel), _A(&A) {}
            /**
             * \brief params[3] out[2]
             */
            bool operator()(const double *const params, double *out) const;
        };

#pragma endregion
    private:
        std::vector<ImgData>
            _data;

        Eigen::Matrix3d _A;

        double _k1;
        double _k2;
        double _k3;

        double _p1;
        double _p2;

        double _theta;

        double _fx;

        double _fy;

    public:
        Camera() = delete;

        /**
         * \brief to load the data
         */
        Camera(std::string mapping_data_dir) { this->init(mapping_data_dir); }

        /**
         * \brief get the data
         */
        const std::vector<ImgData> &data() const { return this->_data; }

        /**
         * \brief do the process
         */
        void process();

        /**
         * \brief get the distortion Params[k1, k2]
         */
        std::tuple<double, double, double> distortionParams() { return std::make_tuple(this->_k1, this->_k2, this->_k3); }

        /**
         * \brief get the inner Params matrix
         */
        const Eigen::Matrix3d &innerParamsMatrix() const { return this->_A; }

    private:
#pragma region private static methods
        /**
         * \brief split a string according the splitor
         */
        static std::vector<std::string> split(const std::string &str, char splitor);
        /**
         * \brief some static private methods
         */
        static void calParams(Camera::ImgData &imgData);

        static void calInitInnerParams(const std::vector<ImgData> &data, double params[6]);

        static Eigen::Matrix3d calOuterParams(const Eigen::Matrix3d &A, const Eigen::Matrix3d &H);

        static Eigen::Matrix3d organizeInnerMatrix(const double params[6]);

        static void doLM(const std::vector<ImgData> &data, double params[5]);

        static void output2Console(const std::string &str);

        static void calDistortionParams(const std::vector<ImgData> &data, const Eigen::Matrix3d &A, double params[2]);

        static void Optimization(const std::vector<ImgData> &data, double innerParams[5], double distortionParams[2]);
#pragma endregion

    private:
#pragma region private member methods
        /**
         * \brief init the class object
         */
        void init(std::string mapping_data_dir);

#pragma endregion
    };
} // namespace ns_test
