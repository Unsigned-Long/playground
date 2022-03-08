#include "handler.h"
#include <fstream>
#include <iomanip>

namespace ns_test
{
#pragma region inner structures
    bool Camera::Ceres_Params::operator()(const double *const params, double *out) const
    {
        Eigen::Matrix3d H;
        for (int i = 0; i != 8; ++i)
            H(i / 3, i % 3) = params[i];
        H(2, 2) = 1.0;
        Eigen::MatrixXd Real(3, 1);
        Real(0, 0) = this->_map->_real.x();
        Real(1, 0) = this->_map->_real.y();
        Real(2, 0) = 1.0;
        auto Pixel = H * Real;
        out[0] = this->_map->_pixel.x() * Pixel(2, 0) - Pixel(0, 0);
        out[1] = this->_map->_pixel.y() * Pixel(2, 0) - Pixel(1, 0);
        return true;
    }

    Eigen::MatrixXd Camera::Ceres_InnerParams::helper(int i, int j) const
    {
        Eigen::MatrixXd V(1, 6);

        /**
         * \brief
         * H1iH1j
         * H1iH2j + H2iH1j
         * H2iH2j
         * H1iH3j + H3iH1j 
         * H2iH3j + H3iH2j
         * H3iH3j
         */
        V(0, 0) = (*_H)(1 - 1, i - 1) * (*_H)(1 - 1, j - 1);
        V(0, 1) = (*_H)(1 - 1, i - 1) * (*_H)(2 - 1, j - 1) + (*_H)(2 - 1, i - 1) * (*_H)(1 - 1, j - 1);
        V(0, 2) = (*_H)(2 - 1, i - 1) * (*_H)(2 - 1, j - 1);
        V(0, 3) = (*_H)(1 - 1, i - 1) * (*_H)(3 - 1, j - 1) + (*_H)(3 - 1, i - 1) * (*_H)(1 - 1, j - 1);
        V(0, 4) = (*_H)(2 - 1, i - 1) * (*_H)(3 - 1, j - 1) + (*_H)(3 - 1, i - 1) * (*_H)(2 - 1, j - 1);
        V(0, 5) = (*_H)(3 - 1, i - 1) * (*_H)(3 - 1, j - 1);
        return V;
    }

    bool Camera::Ceres_InnerParams::operator()(const double *const params, double *out) const
    {
        Eigen::MatrixXd b(6, 1);
        for (int i = 0; i != 6; ++i)
            b(i, 0) = params[i];
        auto v12 = this->helper(1, 2);
        auto v11 = this->helper(1, 1);
        auto v22 = this->helper(2, 2);
        out[0] = 0.0 - (v12 * b)(0, 0);
        out[1] = 0.0 - ((v11 - v22) * b)(0, 0);
        return true;
    }

    bool Camera::Ceres_ML::operator()(const double *const innerParams, const double *const outerParams, double *out) const
    {
        Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
        A(0, 0) = innerParams[0];
        A(1, 1) = innerParams[1];
        A(0, 1) = innerParams[2];
        A(0, 2) = innerParams[3];
        A(1, 2) = innerParams[4];

        auto Rt = Eigen::Matrix3d(outerParams);

        Eigen::MatrixXd M(3, 1);
        M(0, 0) = this->_map->_real.x();
        M(1, 0) = this->_map->_real.y();
        M(2, 0) = 1.0;
        auto m = A * Rt * M;
        Point2d p(m(0, 0) / m(2, 0), m(1, 0) / m(2, 0));

        out[0] = this->_map->_pixel.x() - p.x();
        out[1] = this->_map->_pixel.y() - p.y();

        return true;
    }

    bool Camera::Ceres_DistortionParams::operator()(const double *const params, double *out) const
    {
        auto pixel = (*this->_A) * (this->_xy);

        double r = std::sqrt(std::pow(this->_xy(0, 0) / this->_xy(2, 0), 2) +
                             std::pow(this->_xy(1, 0) / this->_xy(2, 0), 2));

        auto val = std::pow(r, 2) * params[0] +
                   std::pow(r, 4) * params[1] +
                   std::pow(r, 6) * params[2];

        Point2d p(pixel(0, 0) / pixel(2, 0), pixel(1, 0) / pixel(2, 0));

        Point2d center((*this->_A)(0, 2), (*this->_A)(1, 2));

        out[0] = this->_disPixel->x() - p.x() -
                 (p.x() - center.x()) * val;

        out[1] = this->_disPixel->y() - p.y() -
                 (p.y() - center.y()) * val;

        return true;
    }

#pragma endregion

#pragma region public member methods
    void Camera::process()
    {
        Camera::output2Console("Calculate Params Matrix");
        // calculate the H matrix for each image data set
        for (auto &elem : this->_data)
            Camera::calParams(elem);

        Camera::output2Console("Calculate Inner Params Matrix");
        // calculate the A matrix for all images
        double params[6];
        Camera::calInitInnerParams(this->_data, params);
        this->_A = Camera::organizeInnerMatrix(params);

        // alpha beta gamma u0 v0
        double innerParams[5];
        innerParams[0] = this->_A(0, 0);
        innerParams[1] = this->_A(1, 1);
        innerParams[2] = this->_A(0, 1);
        innerParams[3] = this->_A(0, 2);
        innerParams[4] = this->_A(1, 2);

        Camera::output2Console("Calculate Outer Params Matrix");
        // calculate the Rt matrix for each image data set
        for (auto &elem : this->_data)
            elem._Rt = Camera::calOuterParams(this->_A, elem._H);

        Camera::output2Console("Do Levenberg-Marquardt");
        // do Levenberg-Marquardt
        this->doLM(this->_data, innerParams);

        // organize the new values
        this->_A(0, 0) = innerParams[0];
        this->_A(1, 1) = innerParams[1];
        this->_A(0, 1) = innerParams[2];
        this->_A(0, 2) = innerParams[3];
        this->_A(1, 2) = innerParams[4];

        double distortionParams[3];
        Camera::output2Console("Calculate Distortion Params");
        Camera::calDistortionParams(this->_data, this->_A, distortionParams);
        this->_k1 = distortionParams[0];
        this->_k2 = distortionParams[1];
        this->_k3 = distortionParams[2];

        auto alpha = this->_A(0, 0);
        auto gamma = this->_A(0, 1);
        auto beta = this->_A(1, 1);
        this->_theta = std::atan(-alpha / gamma);
        this->_fx = alpha;
        this->_fy = beta * std::sin(this->_theta);

        Camera::output2Console("Finished");

        // std::fixed(std::cout);
        std::cout << std::setiosflags(std::ios::scientific) << std::setprecision(5);

        Camera::output2Console("Inner Params Matrix");
        std::cout << this->_A << std::endl;
        Camera::output2Console("Details");
        std::cout << std::setw(8) << "cx : " << std::setw(12) << this->_A(0, 2) << std::endl;
        std::cout << std::setw(8) << "cy : " << std::setw(12) << this->_A(1, 2) << std::endl;
        std::cout << std::setw(8) << "fx : " << std::setw(12) << this->_fx << std::endl;
        std::cout << std::setw(8) << "fy : " << std::setw(12) << this->_fy << std::endl;
        std::cout << std::setw(8) << "Theta : " << std::setw(12) << this->_theta << std::endl;
        std::cout << std::setw(8) << "K1 : " << std::setw(12) << this->_k1 << std::endl;
        std::cout << std::setw(8) << "K2 : " << std::setw(12) << this->_k2 << std::endl;
        std::cout << std::setw(8) << "K3 : " << std::setw(12) << this->_k3 << std::endl;
    }

#pragma endregion

#pragma region private member methods
    void Camera::init(std::string mapping_data_dir)
    {
        Camera::output2Console("Load Data");
        if (mapping_data_dir.back() != '/')
            mapping_data_dir.push_back('/');
        std::string cmd = "ls " + mapping_data_dir + " > ./fileList.txt";
        auto r = std::system(cmd.c_str());
        std::fstream file("./fileList.txt", std::ios::in);
        if (!file.is_open())
            exit(-1);
        std::vector<std::string> paths;
        std::string strLine;
        while (std::getline(file, strLine))
            paths.push_back(mapping_data_dir + strLine);
        file.close();
        cmd = "rm ./fileList.txt";
        r = std::system(cmd.c_str());
        for (const auto &filePath : paths)
        {
            file.open(filePath, std::ios::in);
            if (!file.is_open())
                exit(-1);
            Camera::Mapping map;
            this->_data.push_back(ImgData());
            while (std::getline(file, strLine))
            {
                auto vec = Camera::split(strLine, ',');
                map._pixel.x() = std::stod(vec.at(0));
                map._pixel.y() = std::stod(vec.at(1));
                map._real.x() = std::stod(vec.at(2));
                map._real.y() = std::stod(vec.at(3));
                this->_data.back()._mappingVec.push_back(map);
            }
            file.close();
        }
        return;
    }
#pragma endregion

#pragma region private static methods

    void Camera::output2Console(const std::string &str)
    {
        std::cout << str << std::endl;
        std::cout << std::string(str.size(), '-') << std::endl;
    }

    std::vector<std::string> Camera::split(const std::string &str, char splitor)
    {
        auto iter = str.begin();
        std::vector<std::string> vec;
        while (true)
        {
            auto pos = std::find(iter, str.end(), splitor);
            vec.push_back(std::string(iter, pos));
            if (pos == str.end())
                break;
            iter = ++pos;
        }
        return vec;
    }

    void Camera::calParams(Camera::ImgData &imgData)
    {
        // the params to calculate
        double params[8];
        // the init values
        std::fill(params, params + 8, 1.0);
        // ceres problems
        ceres::Problem prob;
        // add data to the ceres problem
        for (const auto &map : imgData._mappingVec)
        {
            ceres::CostFunction *fun =
                new ceres::NumericDiffCostFunction<Camera::Ceres_Params, ceres::CENTRAL, 2, 8>(new Camera::Ceres_Params(&map));

            prob.AddResidualBlock(fun, nullptr, params);
        }

        // set solve options
        ceres::Solver::Options op;
        op.gradient_tolerance = 1E-20;
        op.function_tolerance = 1E-20;
        op.minimizer_progress_to_stdout = false;
        op.linear_solver_type = ceres::DENSE_QR;

        // the solve summary
        ceres::Solver::Summary s;

        // solve ceres problem
        ceres::Solve(op, &prob, &s);

        // assign
        for (int i = 0; i != 8; ++i)
            imgData._H(i / 3, i % 3) = params[i];
        imgData._H(2, 2) = 1.0;
        return;
    }

    Eigen::Matrix3d Camera::organizeInnerMatrix(const double params[6])
    {
        Eigen::Matrix3d B;
        B(1 - 1, 1 - 1) = params[0];
        B(1 - 1, 2 - 1) = B(2 - 1, 1 - 1) = params[1];
        B(2 - 1, 2 - 1) = params[2];
        B(1 - 1, 3 - 1) = B(3 - 1, 1 - 1) = params[3];
        B(2 - 1, 3 - 1) = B(3 - 1, 2 - 1) = params[4];
        B(3 - 1, 3 - 1) = params[5];

        // organize the inner matrix
        auto v0 = (B(1 - 1, 2 - 1) * B(1 - 1, 3 - 1) - B(1 - 1, 1 - 1) * B(2 - 1, 3 - 1)) /
                  (B(1 - 1, 1 - 1) * B(2 - 1, 2 - 1) - B(1 - 1, 2 - 1) * B(1 - 1, 2 - 1));
        auto lambda = B(3 - 1, 3 - 1) -
                      (B(1 - 1, 3 - 1) * B(1 - 1, 3 - 1) + v0 * (B(1 - 1, 2 - 1) * B(1 - 1, 3 - 1) - B(1 - 1, 1 - 1) * B(2 - 1, 3 - 1))) / B(1 - 1, 1 - 1);
        auto alpha = std::sqrt(lambda / B(1 - 1, 1 - 1));
        auto beta = std::sqrt(lambda * B(1 - 1, 1 - 1) / (B(1 - 1, 1 - 1) * B(2 - 1, 2 - 1) - B(1 - 1, 2 - 1) * B(1 - 1, 2 - 1)));
        auto gamma = -B(1 - 1, 2 - 1) * alpha * alpha * beta / lambda;
        auto u0 = gamma * v0 / beta - B(1 - 1, 3 - 1) * alpha * alpha / lambda;

        Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
        A(0, 0) = alpha;
        A(1, 1) = beta;
        A(0, 1) = gamma;
        A(0, 2) = u0;
        A(1, 2) = v0;

        return A;
    }

    void Camera::calInitInnerParams(const std::vector<ImgData> &data, double params[6])
    {
        // the init values
        std::fill(params, params + 6, 1.0);
        // ceres problems
        ceres::Problem prob;
        // add data to the ceres problem
        for (const auto &elem : data)
        {
            ceres::CostFunction *fun =
                new ceres::NumericDiffCostFunction<Camera::Ceres_InnerParams, ceres::CENTRAL, 2, 6>(new Ceres_InnerParams(&elem._H));
            prob.AddResidualBlock(fun, nullptr, params);
        }

        // set solve options
        ceres::Solver::Options op;
        op.gradient_tolerance = 1E-20;
        op.function_tolerance = 1E-20;
        op.minimizer_progress_to_stdout = false;
        op.linear_solver_type = ceres::DENSE_QR;

        // the solve summary
        ceres::Solver::Summary s;

        // solve ceres problem
        ceres::Solve(op, &prob, &s);
    }

    Eigen::Matrix3d Camera::calOuterParams(const Eigen::Matrix3d &A, const Eigen::Matrix3d &H)
    {
        auto M = A.inverse() * H;

        Eigen::Vector3d r1 = M.col(0);
        Eigen::Vector3d r2 = M.col(1);
        Eigen::Vector3d t = M.col(2);

        auto lambda = (r1.norm() + r2.norm()) / 2.0;

        r1.normalize();
        r2.normalize();

        Eigen::Matrix3d Rt = Eigen::Matrix3d::Identity();

        for (int i = 0; i != 3; ++i)
        {
            Rt(i, 0) = r1(i, 0);
            Rt(i, 1) = r2(i, 0);
            Rt(i, 2) = t(i, 0) / lambda;
        }
        return Rt;
    }

    void Camera::doLM(const std::vector<ImgData> &data, double params[5])
    {
        ceres::Problem prob;
        for (auto &imgData : data)
        {
            for (auto &elem : imgData._mappingVec)
            {
                ceres::CostFunction *fun =
                    new ceres::NumericDiffCostFunction<Camera::Ceres_ML, ceres::CENTRAL, 2, 5, 9>(new Ceres_ML(&elem));
                prob.AddResidualBlock(fun, nullptr, params, const_cast<double *>(imgData._Rt.data()));
            }
        }

        // set solve options
        ceres::Solver::Options op;
        op.gradient_tolerance = 1E-20;
        op.function_tolerance = 1E-20;
        op.minimizer_progress_to_stdout = false;
        op.linear_solver_type = ceres::DENSE_QR;

        // the solve summary
        ceres::Solver::Summary s;

        // solve ceres problem
        ceres::Solve(op, &prob, &s);

        return;
    }

    void Camera::calDistortionParams(const std::vector<ImgData> &data, const Eigen::Matrix3d &A, double params[2])
    {
        params[0] = 0.0;
        params[1] = 0.0;
        params[2] = 0.0;

        // ceres problems
        ceres::Problem prob;
        // add data to the ceres problem
        for (auto &imgData : data)
        {
            for (auto &elem : imgData._mappingVec)
            {
                Eigen::MatrixXd M(3, 1);
                M(0, 0) = elem._real.x();
                M(1, 0) = elem._real.y();
                M(2, 0) = 1.0;

                auto xy = imgData._Rt * M;

                ceres::CostFunction *fun =
                    new ceres::NumericDiffCostFunction<Camera::Ceres_DistortionParams, ceres::CENTRAL, 2, 3>(new Ceres_DistortionParams(xy, elem._pixel, A));
                prob.AddResidualBlock(fun, nullptr, params);
            }
        }

        // set solve options
        ceres::Solver::Options op;
        op.gradient_tolerance = 1E-20;
        op.function_tolerance = 1E-20;
        op.minimizer_progress_to_stdout = false;
        op.linear_solver_type = ceres::DENSE_QR;

        // the solve summary
        ceres::Solver::Summary s;

        // solve ceres problem
        ceres::Solve(op, &prob, &s);
    }
#pragma endregion
} // namespace ns_test
